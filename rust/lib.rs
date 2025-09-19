// Local library modules.
pub mod index;
pub mod search;
pub mod utils;

// External crate imports.
use anyhow::anyhow;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::{Device, Kind, Tensor};

// Standard library imports.
use std::ffi::CString;

// Conditional imports for cross-platform dynamic library loading.
#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::libloaderapi::LoadLibraryA;

// Internal module imports.
use crate::index::create::create_index;
use crate::index::update::update_index;
use search::load::{load_index, LoadedIndex};
use search::search::{search_index, QueryResult, SearchParameters};

/// Helper function to convert a 3D tensor (batch of queries) into a vector of 2D tensors.
/// This is useful for converting from the old API format to the new format.
fn tensor_to_query_list(queries_tensor: &Tensor) -> Vec<Tensor> {
    let num_queries = queries_tensor.size()[0];
    let mut query_list = Vec::with_capacity(num_queries as usize);
    
    for i in 0..num_queries {
        query_list.push(queries_tensor.get(i));
    }
    
    query_list
}

/// A FastPlaid index that can be loaded once and searched multiple times.
///
/// This class represents a loaded index and provides methods to search it
/// efficiently. The index is loaded once and can be searched multiple times
/// without reloading, making it ideal for serving multiple queries.
#[pyclass(unsendable)]
pub struct Index {
    loaded_index: LoadedIndex,
    device: Device,
}

#[pymethods]
impl Index {
    /// Creates a new Index by loading from disk.
    ///
    /// Args:
    ///     index_path (str): Path to the directory containing the pre-built index.
    ///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
    ///     device (str): The compute device for the search (e.g., "cpu", "cuda:0").
    ///
    /// Returns:
    ///     Index: A loaded index ready for searching.
    #[staticmethod]
    fn load(
        _py: Python<'_>,
        index_path: String,
        torch_path: String,
        device: String,
    ) -> PyResult<Self> {
        call_torch(torch_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

        let device = get_device(&device)?;
        let loaded_index = load_index(&index_path, device).map_err(anyhow_to_pyerr)?;

        Ok(Index {
            loaded_index,
            device,
        })
    }

    /// Searches the loaded index with the given query embeddings.
    ///
    /// Args:
    ///     queries_embeddings (list[torch.Tensor]): A list of tensors, where each tensor
    ///         represents the embeddings for a single query.
    ///     search_parameters (SearchParameters): A configuration object specifying
    ///         search behavior, such as `k` and `nprobe`.
    ///     show_progress (bool): Whether to show a progress bar during search.
    ///     subset (list[list[int]], optional): A list where each inner list contains
    ///         the document IDs to restrict the search to for the corresponding query.
    ///
    /// Returns:
    ///     list[QueryResult]: A list of result objects, each containing the
    ///     `doc_id` and `score` for a retrieved document.
    fn search(
        &self,
        _py: Python<'_>,
        queries_embeddings: Vec<PyTensor>,
        search_parameters: &SearchParameters,
        show_progress: bool,
        subset: Option<Vec<Vec<i64>>>,
    ) -> PyResult<Vec<QueryResult>> {
        let query_tensors: Vec<_> = queries_embeddings
            .into_iter()
            .map(|tensor| tensor.to_device(Device::Cpu).to_kind(Kind::Half))
            .collect();

        let results = search_index(
            &query_tensors,
            &self.loaded_index,
            search_parameters,
            self.device,
            show_progress,
            subset,
        )
        .map_err(anyhow_to_pyerr)?;

        Ok(results)
    }
}

/// Converts a Rust `anyhow::Error` into a Python-compatible `PyErr`.
///
/// This utility function acts as a bridge between Rust's flexible `anyhow` error
/// handling and the error types expected by the PyO3 framework.
fn anyhow_to_pyerr(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Dynamically loads the native Torch shared library (e.g., `libtorch.so` or `torch.dll`).
///
/// This is a workaround to ensure Torch's symbols are available in memory,
/// which can prevent linking errors when `tch-rs` is used within a
/// Python extension module.
fn call_torch(torch_path: String) -> Result<(), anyhow::Error> {
    let torch_path_cstr = CString::new(torch_path.clone())
        .map_err(|e| anyhow!("Failed to create CString for libtorch path: {}", e))?;

    #[cfg(unix)]
    {
        let handle = unsafe { libc::dlopen(torch_path_cstr.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return Err(anyhow!(
                "Failed to load Torch library '{}' via dlopen. Check the path and permissions.",
                torch_path
            ));
        }
    }

    #[cfg(windows)]
    {
        let handle = unsafe { LoadLibraryA(torch_path_cstr.as_ptr()) };
        if handle.is_null() {
            let error_code = unsafe { GetLastError() };
            return Err(anyhow!(
                "Failed to load Torch library '{}' via LoadLibraryA. Windows error code: {}",
                torch_path,
                error_code
            ));
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        return Err(anyhow!(
            "Dynamic library loading is not supported on this operating system."
        ));
    }

    Ok(())
}

/// Parses a string identifier into a `tch::Device`.
///
/// Supports simple device strings like "cpu", "cuda", and indexed CUDA devices
/// such as "cuda:0".
fn get_device(device: &str) -> Result<Device, PyErr> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)), // Default to the first CUDA device.
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<usize>().map(Device::Cuda).map_err(|_| {
                    PyValueError::new_err(format!("Invalid CUDA device index: '{}'", parts[1]))
                })
            } else {
                Err(PyValueError::new_err(
                    "Invalid CUDA device format. Expected 'cuda:N'.",
                ))
            }
        },
        _ => Err(PyValueError::new_err(format!(
            "Unsupported device string: '{}'",
            device
        ))),
    }
}

/// Pre-loads the native Torch library from a specified path.
///
/// Call this function once at the start of a Python script if you encounter
/// linking issues with the Torch library, which can occur in complex deployment
/// environments.
///
/// Args:
///     torch_path (str): The file path to the Torch shared library,
///         e.g., `/path/to/libtorch_cuda.so`.
#[pyfunction]
fn initialize_torch(_py: Python<'_>, torch_path: String) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize Torch: {}", e)))
}

/// Creates and saves a new FastPlaid index to disk.
///
/// This function processes document embeddings, clusters them using the provided
/// centroids, calculates quantization residuals, and serializes the complete
/// index structure to the specified directory.
///
/// Args:
///     index (str): The directory path where the new index will be saved.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device to use for index creation (e.g., "cpu", "cuda:0").
///     embedding_dim (int): The dimensionality of the embeddings.
///     nbits (int): The number of bits to use for residual quantization.
///     embeddings (list[torch.Tensor]): A list of 2D tensors, where each tensor
///         is a batch of document embeddings.
///     centroids (torch.Tensor): A 2D tensor of shape `[num_centroids, embedding_dim]`
///         used for vector quantization.
#[pyfunction]
fn create(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embedding_dim: i64,
    nbits: i64,
    embeddings: Vec<PyTensor>,
    centroids: PyTensor,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;
    let centroids = centroids.to_device(device).to_kind(Kind::Half);

    let embeddings: Vec<_> = embeddings
        .into_iter()
        .map(|tensor| tensor.to_device(device).to_kind(Kind::Half))
        .collect();

    create_index(&embeddings, &index, embedding_dim, nbits, device, centroids)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index: {}", e)))
}

/// Updates an existing FastPlaid index with new documents.
///
/// This function loads the configuration from an existing index, processes the
/// new document embeddings, and adds them to the index without rebuilding it
/// from scratch.
///
/// Args:
///     index (str): The directory path of the index to update.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device to use (e.g., "cpu", "cuda:0").
///     embeddings (list[torch.Tensor]): A list of 2D tensors containing the
///         new document embeddings to add to the index.
#[pyfunction]
fn update(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embeddings: Vec<PyTensor>,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;

    let embeddings: Vec<_> = embeddings
        .into_iter()
        .map(|tensor| tensor.to_device(device).to_kind(Kind::Half))
        .collect();

    update_index(&embeddings, &index, device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to update index: {}", e)))
}

/// Loads an index from disk and returns it for repeated searching.
///
/// This function loads an index and returns a handle that can be used
/// for multiple search operations without reloading the index.
///
/// Args:
///     index_path (str): Path to the directory containing the pre-built index.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device for the search (e.g., "cpu", "cuda:0").
///
/// Returns:
///     Index: A loaded index ready for searching.
#[pyfunction]
fn load(
    _py: Python<'_>,
    index_path: String,
    torch_path: String,
    device: String,
) -> PyResult<Index> {
    Index::load(_py, index_path, torch_path, device)
}

/// Loads an index and performs a search in a single, high-level operation.
///
/// This is a convenience function that combines loading and searching.
/// For multiple searches on the same index, prefer using the Index class
/// with separate load() and search() calls.
///
/// Args:
///     index (str): Path to the directory containing the pre-built index.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device for the search (e.g., "cpu", "cuda:0").
///     queries_embeddings (list[torch.Tensor]): A list of tensors, where each tensor
///         represents the embeddings for a single query.
///     search_parameters (SearchParameters): A configuration object specifying
///         search behavior, such as `k` and `nprobe`.
///     subset (list[list[int]], optional): A list where each inner list contains
///         the document IDs to restrict the search to for the corresponding query.
///
/// Returns:
///     list[QueryResult]: A list of result objects, each containing the
///     `doc_id` and `score` for a retrieved document.
#[pyfunction]
fn load_and_search(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    queries_embeddings: Vec<PyTensor>,
    search_parameters: &SearchParameters,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
) -> PyResult<Vec<QueryResult>> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let query_tensors: Vec<_> = queries_embeddings
        .into_iter()
        .map(|tensor| tensor.to_device(Device::Cpu).to_kind(Kind::Half))
        .collect();

    let device = get_device(&device)?;
    let loaded_index = load_index(&index, device).map_err(anyhow_to_pyerr)?;

    let results = search_index(
        &query_tensors,
        &loaded_index,
        search_parameters,
        device,
        show_progress,
        subset,
    )
    .map_err(anyhow_to_pyerr)?;

    Ok(results)
}

/// Loads an index and performs a search with a single 3D tensor (backward compatibility).
///
/// This function provides backward compatibility for users who have a single 3D tensor
/// with shape [num_queries, tokens_per_query, dim]. It converts the tensor to the new
/// format and calls the main search function.
///
/// Args:
///     index (str): Path to the directory containing the pre-built index.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device for the search (e.g., "cpu", "cuda:0").
///     queries_tensor (torch.Tensor): A 3D tensor of query embeddings with
///         shape `[num_queries, tokens_per_query, dim]`.
///     search_parameters (SearchParameters): A configuration object specifying
///         search behavior, such as `k` and `nprobe`.
///     subset (list[list[int]], optional): A list where each inner list contains
///         the document IDs to restrict the search to for the corresponding query.
///
/// Returns:
///     list[QueryResult]: A list of result objects, each containing the
///     `doc_id` and `score` for a retrieved document.
#[pyfunction]
fn load_and_search_tensor(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    queries_tensor: PyTensor,
    search_parameters: &SearchParameters,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
) -> PyResult<Vec<QueryResult>> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let queries_tensor = queries_tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Half);
    
    // Convert 3D tensor to vector of 2D tensors
    let query_tensors = tensor_to_query_list(&queries_tensor);

    let device = get_device(&device)?;
    let loaded_index = load_index(&index, device).map_err(anyhow_to_pyerr)?;

    let results = search_index(
        &query_tensors,
        &loaded_index,
        search_parameters,
        device,
        show_progress,
        subset,
    )
    .map_err(anyhow_to_pyerr)?;

    Ok(results)
}

/// A high-performance document retrieval toolkit using a ColBERT-style late
/// interaction model, implemented in Rust with Python bindings.
///
/// This module provides functions for creating, updating, and searching indexes,
/// along with the necessary data classes `SearchParameters`, `QueryResult`, and
/// `Index` to interact with the library from Python.
#[pymodule]
#[pyo3(name = "fast_plaid_rust")]
fn python_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add data classes required for the Python interface.
    m.add_class::<SearchParameters>()?;
    m.add_class::<QueryResult>()?;
    m.add_class::<Index>()?;

    // Add functions to the Python module.
    m.add_function(wrap_pyfunction!(initialize_torch, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(update, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_search, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_search_tensor, m)?)?;

    Ok(())
}
