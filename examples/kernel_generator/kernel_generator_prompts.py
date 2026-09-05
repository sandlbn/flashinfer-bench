"""
This file contains the prompts for baseline agent generation.
"""

from flashinfer_bench import FFI_PROMPT_SIMPLE, SYCL_PROMPT_SIMPLE, Definition, EvaluationStatus, Trace


def _format_definition(definition: Definition) -> str:
    axes_str = "\nAxes:\n"
    for name, axis in definition.axes.items():
        if hasattr(axis, "value"):
            axes_str += f"  {name}: constant = {axis.value}"
        else:
            axes_str += f"  {name}: variable"
        if axis.description:
            axes_str += f" ({axis.description})"
        axes_str += "\n"

    # Format inputs
    inputs_str = "\nInputs:\n"
    for name, spec in definition.inputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        inputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            inputs_str += f" - {spec.description}"
        inputs_str += "\n"

    outputs_str = "\nOutputs:\n"
    for name, spec in definition.outputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        outputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            outputs_str += f" - {spec.description}"
        outputs_str += "\n"

    constraints_str = ""
    if definition.constraints:
        constraints_str = "\nConstraints:\n"
        for constraint in definition.constraints:
            constraints_str += f"  - {constraint}\n"

    return f"""Name: {definition.name}
Type: {definition.op_type}
{axes_str}{inputs_str}{outputs_str}{constraints_str}

Reference Implementation:
{definition.reference}"""


def _format_trace_logs(trace: Trace) -> str:
    if trace.is_workload_trace() or not trace.evaluation:
        return "No evaluation logs available (workload-only trace)"

    eval_info = f"Status: {trace.evaluation.status.value}\n"
    eval_info += f"Timestamp: {trace.evaluation.timestamp}\n"

    if trace.evaluation.log:
        eval_info += f"\nExecution Log:\n{trace.evaluation.log}\n"

    if trace.evaluation.correctness:
        eval_info += f"Max relative error: {trace.evaluation.correctness.max_relative_error}\n"
        eval_info += f"Max absolute error: {trace.evaluation.correctness.max_absolute_error}\n"

    if trace.evaluation.performance:
        eval_info += f"Latency: {trace.evaluation.performance.latency_ms}ms\n"
        eval_info += f"Reference latency: {trace.evaluation.performance.reference_latency_ms}ms\n"
        eval_info += f"Speedup factor: {trace.evaluation.performance.speedup_factor}x\n"

    return eval_info


TRITON_PROMPT = """Generate a Triton kernel optimized for {target_gpu} GPU for

{definition}

Triton Version: 3.3.1

Requirements:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your Triton implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the code, no explanations or markdown formatting

Generate complete, runnable code only - no framework will add device handling wrapper code.

Generate the implementation:"""

TRITON_OPTIMIZATION_PROMPT = """You are optimizing a Triton kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations
   - Ensure numerical correctness matches the reference implementation

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns for {target_gpu}
   - Tune block sizes and grid dimensions
   - Use appropriate Triton language features for vectorization
   - Minimize global memory transactions

Requirements for the optimized implementation:
- Write clean, efficient Triton code optimized for {target_gpu} architecture
- Use modern Triton syntax with proper grid computation and language features
- Include necessary imports (torch, triton, triton.language as tl)
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signature and device handling as specified

The wrapper function MUST handle complete device management:
- Move CPU tensors to GPU if needed (use .cuda() when torch.cuda.is_available())
- Raise clear errors if CUDA is not available for GPU tensors
- Call the triton kernel with GPU tensors
- Move results back to original device of input tensors
- Handle both args and kwargs properly
- Preserve original tensor devices and restore them for outputs

IMPORTANT: Use only valid Python/Triton syntax:
- NO hexadecimal float literals (0x1.234p5) - use decimal equivalents
- NO C/CUDA specific syntax - this is Python/Triton code
- All code must be valid Python that passes ast.parse()

- Expose a "run" entry point function that can be called to execute the kernel
- Return only the improved code, no explanations or markdown formatting

Generate the corrected and optimized implementation:"""

PYTHON_PROMPT = """You are a code generator. Generate a Python implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient Python code optimized for {target_gpu} architecture
- Use PyTorch operations when appropriate, optimized for {target_gpu}
- Include necessary imports
- Implement the exact functionality described in the specification
- Expose a "run" entry point function that can be called to execute the implementation
- Return only the code, no explanations or markdown formatting

Generate the implementation:"""

CUDA_PROMPT = """You are a code generator. Generate a CUDA kernel implementation optimized for {target_gpu} GPU for the following specification.

Specification:
{definition}

Requirements:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and memory management optimized for {target_gpu}
- Implement the exact functionality described in the specification
- The reference code provides the mathematical specification but is unoptimized - your CUDA implementation should match its computational accuracy while delivering high performance
- Use the definition's tensor shapes, dtypes, and axes information to guide memory access patterns and optimization strategies
- Optimize for {target_gpu} GPU characteristics (memory hierarchy, compute units, etc.)
- For fixed axis values, optimize specifically for those constants rather than general cases

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Ensure proper CUDA stream synchronization and error handling

Generate the implementation:"""

CUDA_OPTIMIZATION_PROMPT = """You are optimizing a CUDA kernel for {target_gpu} GPU. The current implementation has issues that need to be fixed.

Original Specification:
{definition}

Current Implementation Status:
{trace_logs}

Current Implementation:
{current_code}

Optimization Strategy:
1. ENSURE CORRECTNESS: If there are compile errors, runtime errors, or incorrect outputs, focus entirely on fixing these issues
   - Analyze compilation errors and fix syntax/API usage
   - Fix runtime errors like shape mismatches, memory access violations, kernel launch failures
   - Ensure numerical correctness matches the reference implementation
   - Verify proper CUDA memory management and synchronization

2. OPTIMIZE PERFORMANCE: if the current kernel is functionally correct, focus on performance optimizations
   - Optimize memory access patterns and coalescing for {target_gpu}
   - Tune block sizes and grid dimensions for maximum occupancy
   - Utilize shared memory effectively to reduce global memory transactions
   - Optimize register usage and minimize divergent branches
   - Consider using specialized libraries (such as CUTLASS) where beneficial
   - Leverage constant axis values for compile-time optimizations

Requirements for the optimized implementation:
- Write clean, efficient CUDA C++ code optimized for {target_gpu} architecture
- Use proper CUDA syntax and modern features appropriate for {target_gpu}
- Fix all identified issues from the feedback
- Maintain or improve computational accuracy
- Preserve the same function signatures and device handling as specified
- For fixed axis values, optimize specifically for those constants rather than general cases

IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
- Proper error checking and memory management
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Memory allocation and data transfer management
- Device management and error handling
- Entry point function named "run" that can be called to execute the implementation
- Handle both args and kwargs properly
- Move CPU data to GPU, execute kernels, and return results to CPU
</cpp_file>

Code Generation Guidelines:
- Use modern CUDA features appropriate for {target_gpu}
- Optimize memory coalescing and reduce bank conflicts
- Utilize shared memory effectively for data reuse
- Consider occupancy and register usage
- Implement proper error checking with cudaGetLastError()
- Use appropriate grid and block dimensions for the problem size
- Leverage constant memory for frequently accessed read-only data
- Ensure proper CUDA stream synchronization and error handling

Generate the corrected and optimized implementation:"""

TORCH_BINDINGS_PROMPT = """
Use TORCH for your generated kernel host function and bindings

Requirements:
- Include all necessary headers (torch/extension.h, kernel.h, etc.)
- Implement the "run" function that:
  * Takes torch::Tensor arguments
  * Validates tensor properties (device, dtype, shape)
  * Extracts raw pointers using .data_ptr<T>()
  * Calls the CUDA kernel with appropriate launch configuration
  * Returns results as torch::Tensor
- Use PYBIND11_MODULE to bind the "run" function:
  * PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  *   m.def("run", &run, "Kernel execution function");
  * }
- Handle both positional args and kwargs properly
- Include proper error messages for invalid inputs

- Use torch::Tensor for all tensor arguments
- Use .device().is_cuda() to check if tensors are on GPU
- Use .dtype() to validate tensor data types
- Use .sizes() or .size(dim) to get tensor dimensions
- Use .data_ptr<float>() or .data_ptr<T>() to get raw pointers
- Call cudaDeviceSynchronize() or cudaGetLastError() for error checking
- Return torch::Tensor from the run function
- Handle exceptions gracefully with proper error messages"""


SYCL_PROMPT = """You are an expert Intel GPU kernel engineer. Write a SYCL kernel that
implements the following operation, targeting {target_gpu}.

{definition}

Requirements:
- SYCL is C++. Emit one file.
- Export the entry point as the symbol `run` using TVM_FFI_DLL_EXPORT_TYPED_FUNC.
- Arguments arrive in Definition order: every input, then every output.
- Validate shapes and dtypes with TVM_FFI_ICHECK before touching data.
- Accumulate in float even when inputs and outputs are half precision.
- Handle the dtypes the Definition declares; reject anything else with a clear error.

Output format - emit exactly one file in this XML wrapper and nothing else:

<sycl_file name="kernel.cpp">
// your SYCL code here
</sycl_file>

The binding rules that follow are not optional; a kernel that ignores them will either
fail to load or corrupt memory.
"""


SYCL_OPTIMIZATION_PROMPT = """You are optimizing a SYCL kernel for {target_gpu}.

{definition}

Current implementation:
{current_code}

Results from the last run:
{trace_logs}

Work in this order:

1. CORRECTNESS FIRST. If the kernel failed to compile, failed at runtime, or produced
   incorrect output, fix that and change nothing else. A fast wrong kernel is worthless -
   the harness will reject it.
   - Compile errors: check SYCL API usage and that every captured value is device-copyable.
   - Incorrect output: check indexing at the boundary, and that accumulation is in float.

2. THEN PERFORMANCE, if the kernel is already correct. What actually moves the needle on
   Intel GPUs, roughly in order:
   - Work-group and sub-group sizing. Sub-group widths are 16 and 32; pin the one you
     depend on with the reqd_sub_group_size attribute. Intel's own tuned Triton kernels
     overwhelmingly use large work-groups - num_warps=32 equivalents - not the small
     CUDA-style blocks you may be used to.
   - Group collectives - reduce_over_group - instead of hand-written shared-memory trees.
   - Vectorised loads for memory-bound kernels; coalesced access across the sub-group.
   - Shared local memory via local_accessor, staying inside the device budget.
   - For GEMM-shaped work, prefer oneMKL or oneDNN over a hand-written kernel. Hand-write
     when fusing operations the libraries do not cover - that is where the real headroom is.

Report the speedup you are aiming for and why the change should produce it.

Output format - emit exactly one file in this XML wrapper and nothing else:

<sycl_file name="kernel.cpp">
// your improved SYCL code here
</sycl_file>
"""


def get_prompt(
    language: str, definition: Definition, target_gpu: str = "H100", use_ffi: bool = True
) -> str:
    prompts = {
        "triton": TRITON_PROMPT,
        "python": PYTHON_PROMPT,
        "cuda": CUDA_PROMPT,
        "sycl": SYCL_PROMPT,
    }

    if language not in prompts:
        raise ValueError(f"Unsupported language: {language}")

    definition_str = _format_definition(definition)
    base_prompt = prompts[language].format(definition=definition_str, target_gpu=target_gpu)

    if language.lower() == "cuda":
        binding_prompt = FFI_PROMPT_SIMPLE if use_ffi else TORCH_BINDINGS_PROMPT
        base_prompt = base_prompt + "\n\n" + binding_prompt
    elif language.lower() == "sycl":
        # Appended, never formatted: it contains C++ braces, which str.format would
        # read as placeholders and reject.
        base_prompt = base_prompt + "\n\n" + SYCL_PROMPT_SIMPLE

    return base_prompt


def get_optimization_prompt(
    language: str,
    definition,
    trace: Trace,
    current_code: str,
    target_gpu: str = "H100",
    use_ffi: bool = True,
) -> str:
    optimization_prompts = {
        "triton": TRITON_OPTIMIZATION_PROMPT,
        "cuda": CUDA_OPTIMIZATION_PROMPT,
        "sycl": SYCL_OPTIMIZATION_PROMPT,
    }

    if language not in optimization_prompts:
        raise ValueError(f"Unsupported language for optimization: {language}")

    definition_str = _format_definition(definition)
    trace_logs = _format_trace_logs(trace)

    base_prompt = optimization_prompts[language].format(
        definition=definition_str,
        trace_logs=trace_logs,
        current_code=current_code,
        target_gpu=target_gpu,
    )

    if language.lower() == "cuda":
        binding_prompt = FFI_PROMPT_SIMPLE if use_ffi else TORCH_BINDINGS_PROMPT
        base_prompt = base_prompt + "\n\n" + binding_prompt
    elif language.lower() == "sycl":
        base_prompt = base_prompt + "\n\n" + SYCL_PROMPT_SIMPLE

    return base_prompt
