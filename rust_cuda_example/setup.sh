#!/bin/bash

echo "🚀 Configurando ambiente Rust + CUDA"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para verificar se o comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para imprimir com cor
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo "=====================================";
echo "1. Verificando dependências..."

# Verificar se Rust está instalado
if command_exists rustc; then
    RUST_VERSION=$(rustc --version)
    print_status "Rust encontrado: $RUST_VERSION"
else
    print_error "Rust não encontrado. Instalando Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    print_status "Rust instalado com sucesso!"
fi

# Verificar se Cargo está funcionando
if command_exists cargo; then
    CARGO_VERSION=$(cargo --version)
    print_status "Cargo encontrado: $CARGO_VERSION"
else
    print_error "Cargo não encontrado!"
    exit 1
fi

echo "=====================================";
echo "2. Verificando CUDA..."

# Verificar se nvidia-smi existe
if command_exists nvidia-smi; then
    print_status "nvidia-smi encontrado"
    nvidia-smi
else
    print_warning "nvidia-smi não encontrado. Você pode não ter uma GPU NVIDIA ou drivers instalados."
    print_info "O programa ainda funcionará com simulação CPU."
fi

# Verificar se nvcc existe
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}')
    print_status "CUDA encontrado: $CUDA_VERSION"
else
    print_warning "CUDA Toolkit não encontrado."
    print_info "Você pode instalar com:"
    echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "  Ou baixe do site oficial da NVIDIA"
    print_info "O programa ainda funcionará com simulação CPU."
fi

echo "=====================================";
echo "3. Configurando projeto Rust..."

# Criar diretório se não existir
if [ ! -d "rust_cuda_example" ]; then
    mkdir rust_cuda_example
    print_status "Diretório 'rust_cuda_example' criado"
fi

cd rust_cuda_example

# Verificar se src existe
if [ ! -d "src" ]; then
    mkdir src
    print_status "Diretório 'src' criado"
fi

echo "=====================================";
echo "4. Verificando variáveis de ambiente CUDA..."

# Verificar CUDA_PATH
if [ -z "$CUDA_PATH" ]; then
    print_warning "CUDA_PATH não definido"
    
    # Tentar encontrar CUDA automaticamente
    POSSIBLE_PATHS=(
        "/usr/local/cuda"
        "/opt/cuda"
        "/usr/cuda"
    )
    
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ]; then
            print_info "CUDA encontrado em: $path"
            echo "export CUDA_PATH=$path" >> ~/.bashrc
            echo "export PATH=\$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
            export CUDA_PATH=$path
            export PATH=$CUDA_PATH/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
            print_status "Variáveis de ambiente CUDA configuradas"
            break
        fi
    done
else
    print_status "CUDA_PATH já definido: $CUDA_PATH"
fi

echo "=====================================";
echo "5. Compilando projeto..."

# Compilar o projeto
print_info "Executando 'cargo build --release'..."
if cargo build --release; then
    print_status "Compilação bem-sucedida!"
else
    print_error "Erro na compilação. Detalhes acima."
    echo ""
    print_info "Possíveis soluções:"
    echo "  1. Verifique se todas as dependências estão instaladas"
    echo "  2. Verifique se o CUDA está corretamente configurado"
    echo "  3. O programa ainda pode funcionar mesmo sem CUDA (modo CPU)"
fi

echo "=====================================";
echo "6. Testando execução..."

print_info "Executando 'cargo run --release'..."
cargo run --release

echo "=====================================";
echo "✨ Configuração concluída!"
echo ""
print_info "Para executar novamente:"
echo "  cd rust_cuda_example"
echo "  cargo run --release"
echo ""
print_info "Para executar apenas testes:"
echo "  cargo test"
echo ""
print_warning "Se houver problemas com CUDA, o programa executará em modo CPU simulado."

nvcc --version
nvidia-smi

sudo apt update
sudo apt install nvidia-cuda-toolkit

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

cargo run --release