// projeto: rust_cuda_example
// file: src/main.rs

 


use cudarc::driver::safe::{CudaDevice, CudaSlice};
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use rayon::prelude::*;

// Kernel CUDA para operações de vetores
const CUDA_KERNEL: &str = r#"
extern "C" __global__ void vector_multiply(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void matrix_multiply(float* a, float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Inicializando processamento CUDA com Rust...");
      
    // Tenta inicializar o dispositivo CUDA
    let device: Arc<CudaDevice> = match CudaDevice::new(0) { // Explicit type annotation
        Ok(dev) => {
            println!("✅ Dispositivo CUDA inicializado com sucesso!");
            Arc::new(dev)
        },
        Err(e) => {
            println!("❌ Erro ao inicializar CUDA: {:?}", e);
            println!("💡 Vamos executar uma versão simulada CPU que demonstra os conceitos:");
            run_cpu_simulation()?;
            return Ok(());
        }
    };
    
    println!("📊 Dispositivo: {}", device.name()?);
    println!("📊 Memória total: {:.2} GB", device.total_memory()? as f64 / (1024.0 * 1024.0 * 1024.0));
    
    // ... (rest of the code remains unchanged)


    // Tenta inicializar o dispositivo CUDA
    let device = match CudaDevice::new(0) {
        Ok(dev) => {
            println!("✅ Dispositivo CUDA inicializado com sucesso!");
            Arc::new(dev)
        },
        Err(e) => {
            println!("❌ Erro ao inicializar CUDA: {:?}", e);
            println!("💡 Vamos executar uma versão simulada CPU que demonstra os conceitos:");
            run_cpu_simulation()?;
            return Ok(());
        }
    };
    
    println!("📊 Dispositivo: {}", device.name()?);
    println!("📊 Memória total: {:.2} GB", device.total_memory()? as f64 / (1024.0 * 1024.0 * 1024.0));
    
    // Compila e carrega os kernels CUDA
    match compile_and_load_kernels(&device) {
        Ok(_) => {
            println!("✅ Kernels CUDA compilados e carregados com sucesso!");
            
            // Executa os exemplos com CUDA
            println!("\n🔢 Exemplo 1: Multiplicação de vetores (CUDA)");
            if let Err(e) = vector_multiply_example(&device) {
                println!("❌ Erro: {:?}", e);
            }
            
            println!("\n➕ Exemplo 2: Adição de vetores (CUDA)");
            if let Err(e) = vector_add_example(&device) {
                println!("❌ Erro: {:?}", e);
            }
            
            println!("\n🔄 Exemplo 3: Multiplicação de matrizes (CUDA)");
            if let Err(e) = matrix_multiply_example(&device) {
                println!("❌ Erro: {:?}", e);
            }
            
            println!("\n⚡ Exemplo 4: Benchmark de performance");
            if let Err(e) = performance_benchmark(&device) {
                println!("❌ Erro: {:?}", e);
            }
        },
        Err(e) => {
            println!("❌ Erro ao compilar kernels: {:?}", e);
            println!("💡 Executando apenas testes básicos de memória...");
            run_basic_memory_tests(&device)?;
        }
    }
    
    println!("\n✨ Processamento concluído!");
    Ok(())
}

fn compile_and_load_kernels(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    let ptx = Ptx::from_src(CUDA_KERNEL);
    device.load_ptx(ptx, "vector_ops", &["vector_multiply", "vector_add", "matrix_multiply"])?;
    Ok(())
}

fn run_cpu_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🖥️  === SIMULAÇÃO CPU (Conceitos GPU) ===");
    
    // Simula operações paralelas usando iterators Rust
    let n = 1_000_000;
    
    println!("\n🔢 Multiplicação de vetores paralela (CPU):");
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    
    let start = std::time::Instant::now();
    let result: Vec<f32> = a.par_iter().zip(b.par_iter())
        .map(|(x, y)| x * y)
        .collect();
    let duration = start.elapsed();
    
    println!("Tamanho: {} elementos", n);
    println!("Tempo: {:.2}ms", duration.as_millis());
    println!("Primeiros 5 resultados: {:?}", &result[0..5]);
    
    println!("\n➕ Adição de vetores paralela (CPU):");
    let start = std::time::Instant::now();
    let result_add: Vec<f32> = a.par_iter().zip(b.par_iter())
        .map(|(x, y)| x + y)
        .collect();
    let duration = start.elapsed();
    
    println!("Tempo: {:.2}ms", duration.as_millis());
    println!("Primeiros 5 resultados: {:?}", &result_add[0..5]);
    
    println!("\n🔄 Multiplicação de matrizes paralela (CPU):");
    let n_matrix = 256;
    let matrix_a: Vec<f32> = (0..n_matrix*n_matrix).map(|i| (i % 10) as f32).collect();
    let matrix_b: Vec<f32> = (0..n_matrix*n_matrix).map(|i| ((i * 3) % 7) as f32).collect();
    
    let start = std::time::Instant::now();
    let _result_matrix = multiply_matrices_parallel(&matrix_a, &matrix_b, n_matrix);
    let duration = start.elapsed();
    
    println!("Matriz {}x{}", n_matrix, n_matrix);
    println!("Tempo: {:.2}ms", duration.as_millis());
    println!("GFLOPS: {:.2}", (2.0 * n_matrix as f64 * n_matrix as f64 * n_matrix as f64) / duration.as_secs_f64() / 1e9);
    
    Ok(())
}

fn multiply_matrices_parallel(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    (0..n*n).into_par_iter().map(|idx| {
        let row = idx / n;
        let col = idx % n;
        let mut sum = 0.0;
        for k in 0..n {
            sum += a[row * n + k] * b[k * n + col];
        }
        sum
    }).collect()
}

fn run_basic_memory_tests(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testes básicos de transferência de memória GPU...");
    
    // Teste básico de alocação e transferência
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("📤 Enviando dados para GPU: {:?}", test_data);
    
    let gpu_buffer: CudaSlice<f32> = device.htod_copy(test_data.clone())?;
    let result = device.dtoh_sync_copy(&gpu_buffer)?;
    
    println!("📥 Dados recebidos da GPU: {:?}", result);
    
    if test_data == result {
        println!("✅ Transferência de memória funcionando!");
    } else {
        println!("❌ Erro na transferência de memória!");
    }
    
    // Teste de performance de transferência
    let large_size = 1_000_000;
    println!("\n🚀 Teste de bandwidth - {} elementos", large_size);
    
    let start = std::time::Instant::now();
    let large_data: Vec<f32> = (0..large_size).map(|i| i as f32).collect();
    let gpu_large: CudaSlice<f32> = device.htod_copy(large_data)?;
    let _result_large = device.dtoh_sync_copy(&gpu_large)?;
    let duration = start.elapsed();
    
    println!("⚡ Tempo: {:.2}ms", duration.as_millis());
    println!("📊 Bandwidth: {:.2} GB/s", 
             (large_size * 4 * 2) as f64 / duration.as_secs_f64() / 1e9);
    
    Ok(())
}

fn vector_multiply_example(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    let n = 1_000_000;
    
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    
    let a_gpu: CudaSlice<f32> = device.htod_copy(a.clone())?;
    let b_gpu: CudaSlice<f32> = device.htod_copy(b.clone())?;
    let mut c_gpu: CudaSlice<f32> = device.alloc_zeros::<f32>(n)?;
    
    let block_size = 256;
    let grid_size = (n + block_size - 1) / block_size;
    
    let start = std::time::Instant::now();
    unsafe {
        let func = device.get_func("vector_ops", "vector_multiply")?;
        func.launch(LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        }, (&a_gpu, &b_gpu, &mut c_gpu, n as i32))?;
    }
    device.synchronize()?;
    let duration = start.elapsed();
    
    let result = device.dtoh_sync_copy(&c_gpu)?;
    
    println!("Tamanho: {} elementos", n);
    println!("Tempo: {:.2}ms", duration.as_millis());
    println!("Primeiros 5 resultados: {:?}", &result[0..5]);
    println!("GFLOPS: {:.2}", n as f64 / duration.as_secs_f64() / 1e9);
    
    // Validação
    let expected: Vec<f32> = (0..5).map(|i| (i * i * 2) as f32).collect();
    if result[0..5] == expected {
        println!("✅ Resultados validados!");
    }
    
    Ok(())
}

fn vector_add_example(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    let n = 5_000_000;
    
    let a: Vec<f32> = (0..n).map(|i| (i % 100) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 50) as f32).collect();
    
    let a_gpu: CudaSlice<f32> = device.htod_copy(a.clone())?;
    let b_gpu: CudaSlice<f32> = device.htod_copy(b.clone())?;
    let mut c_gpu: CudaSlice<f32> = device.alloc_zeros::<f32>(n)?;
    
    let block_size = 512;
    let grid_size = (n + block_size - 1) / block_size;
    
    let start = std::time::Instant::now();
    unsafe {
        let func = device.get_func("vector_ops", "vector_add")?;
        func.launch(LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        }, (&a_gpu, &b_gpu, &mut c_gpu, n as i32))?;
    }
    device.synchronize()?;
    let duration = start.elapsed();
    
    let result = device.dtoh_sync_copy(&c_gpu)?;
    
    println!("Tamanho: {} elementos", n);
    println!("Tempo: {:.2}ms", duration.as_millis());
    println!("Bandwidth: {:.2} GB/s", (3 * n * 4) as f64 / duration.as_secs_f64() / 1e9);
    
    Ok(())
}

fn matrix_multiply_example(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    let n = 512;
    let size = n * n;
    
    let a: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| ((i * 3) % 7) as f32).collect();
    
    let a_gpu: CudaSlice<f32> = device.htod_copy(a.clone())?;
    let b_gpu: CudaSlice<f32> = device.htod_copy(b.clone())?;
    let mut c_gpu: CudaSlice<f32> = device.alloc_zeros::<f32>(size)?;
    
    let block_size = 16;
    let grid_size = (n + block_size - 1) / block_size;
    
    let start = std::time::Instant::now();
    unsafe {
        let func = device.get_func("vector_ops", "matrix_multiply")?;
        func.launch(LaunchConfig {
            grid_dim: (grid_size as u32, grid_size as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        }, (&a_gpu, &b_gpu, &mut c_gpu, n as i32))?;
    }
    device.synchronize()?;
    let duration = start.elapsed();
    
    let result = device.dtoh_sync_copy(&c_gpu)?;
    
    println!("Matriz: {}x{}", n, n);
    println!("Tempo: {:.2}ms", duration.as_millis());
    println!("GFLOPS: {:.2}", (2.0 * n as f64 * n as f64 * n as f64) / duration.as_secs_f64() / 1e9);
    
    Ok(())
}

fn performance_benchmark(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmark CPU vs GPU:");
    
    let sizes = vec![100_000, 1_000_000];
    
    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        
        // CPU
        let start_cpu = std::time::Instant::now();
        let _cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        let cpu_duration = start_cpu.elapsed();
        
        // GPU
        let start_gpu = std::time::Instant::now();
        let a_gpu: CudaSlice<f32> = device.htod_copy(a.clone())?;
        let b_gpu: CudaSlice<f32> = device.htod_copy(b.clone())?;
        let mut c_gpu: CudaSlice<f32> = device.alloc_zeros::<f32>(size)?;
        
        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;
        
        unsafe {
            let func = device.get_func("vector_ops", "vector_multiply")?;
            func.launch(LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            }, (&a_gpu, &b_gpu, &mut c_gpu, size as i32))?;
        }
        device.synchronize()?;
        let gpu_duration = start_gpu.elapsed();
        
        println!("\n📊 {} elementos:", size);
        println!("CPU: {:.2}ms", cpu_duration.as_millis());
        println!("GPU: {:.2}ms", gpu_duration.as_millis());
        if gpu_duration.as_millis() > 0 {
            println!("Speedup: {:.2}x", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_initialization() {
        match CudaDevice::new(0) {
            Ok(_) => println!("✅ CUDA disponível"),
            Err(e) => println!("⚠️ CUDA indisponível: {:?}", e),
        }
    }
    
    #[test]
    fn test_cpu_simulation() {
        let result = run_cpu_simulation();
        assert!(result.is_ok());
    }
}




// cd rust_cuda_example

// cargo run --release



// # Ubuntu/Debian
// sudo apt install nvidia-cuda-toolkit

// # Ou baixe do site oficial da NVIDIA








// # Verificar CUDA
// nvcc --version
// nvidia-smi

// # Se não tiver CUDA, instalar:
// sudo apt update
// sudo apt install nvidia-cuda-toolkit

// # Verificar variáveis de ambiente
// echo $CUDA_PATH
// echo $LD_LIBRARY_PATH










// # Ubuntu/Debian
// wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
// sudo dpkg -i cuda-keyring_1.0-1_all.deb
// sudo apt-get update
// sudo apt-get -y install cuda-toolkit-12-0

// # Adicionar ao PATH
// echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
// echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
// source ~/.bashrc