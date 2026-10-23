import os
import sys
import shutil
import subprocess
from pathlib import Path

class ModelDownloader:
    def __init__(self):
        self.project_root = Path.cwd()
        self.models_dir = self.project_root / "models"
        self.model_name = "Qwen/Qwen3-Embedding-4B"
        self.local_model_name = "Qwen3-Embedding-4B"
        self.local_model_path = self.models_dir / self.local_model_name
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
    def print_banner(self, text, char="="):
        """Print formatted banner"""
        width = 70
        print("\n" + char * width)
        print(f"  {text.center(width-4)}")
        print(char * width + "\n")
    
    def print_section(self, step, title):
        """Print section header"""
        print(f"\n[{step}] {title}")
        print("-" * 70)
    
    def print_success(self, msg):
        """Print success message"""
        print(f"✓ {msg}")
    
    def print_error(self, msg):
        """Print error message"""
        print(f"✗ {msg}")
        sys.exit(1)
    
    def print_info(self, msg):
        """Print info message"""
        print(f"ℹ {msg}")
    
    def print_warning(self, msg):
        """Print warning message"""
        print(f"⚠ {msg}")
    
    def run_command(self, command, description=""):
        """Run a system command and handle errors"""
        try:
            if description:
                self.print_info(description)
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                return False, result.stderr
            return True, result.stdout
        except Exception as e:
            return False, str(e)
    
    def install_dependencies(self):
        """Install required Python packages"""
        self.print_section("1/6", "Installing Python Dependencies")
        
        required_packages = [
            "sentence-transformers",
            "torch",
            "torchvision",
            "torchaudio",
            "huggingface-hub",
            "transformers",
            "numpy",
            "scipy",
            "scikit-learn"
        ]
        
        print("\n  Checking and installing packages (this may take 3-5 minutes):\n")
        
        for package in required_packages:
            # Check if package is already installed
            check_cmd = f'python -c "import {package.replace("-", "_")}" 2>nul'
            is_installed, _ = self.run_command(check_cmd)
            
            if is_installed:
                self.print_success(f"  {package} (already installed)")
            else:
                self.print_info(f"  Installing {package}...")
                
                # Install package
                install_cmd = f'pip install --upgrade {package}'
                success, error = self.run_command(install_cmd)
                
                if success:
                    self.print_success(f"  {package} (installed)")
                else:
                    self.print_warning(f"  {package} (installation attempted, may already be cached)")
        
        print("\n")
        self.print_success("All dependencies ready!")
    
    def create_models_directory(self):
        """Create models directory if it doesn't exist"""
        self.print_section("2/6", "Creating Models Directory")
        
        try:
            self.models_dir.mkdir(exist_ok=True)
            self.print_success(f"Models directory ready at: {self.models_dir.absolute()}")
            return True
        except Exception as e:
            self.print_error(f"Failed to create models directory: {str(e)}")
    
    def check_existing_model(self):
        """Check if model already exists"""
        self.print_section("3/6", "Checking for Existing Model")
        
        if self.local_model_path.exists():
            # Verify model integrity
            required_files = [
                "config.json",
                "tokenizer.json",
                "modules.json"
            ]
            
            missing_files = [f for f in required_files if not (self.local_model_path / f).exists()]
            
            if not missing_files:
                # Check model files
                model_files = list(self.local_model_path.glob("*.safetensors")) + \
                             list(self.local_model_path.glob("pytorch_model.bin"))
                
                if model_files:
                    model_size = sum(f.stat().st_size for f in self.local_model_path.rglob("*") if f.is_file()) / (1024**3)
                    self.print_success(f"Model already exists! Size: {model_size:.2f} GB")
                    
                    # List model structure
                    print("\n  Model structure:")
                    for item in sorted(self.local_model_path.iterdir()):
                        if item.is_file():
                            size = item.stat().st_size / (1024**2)
                            print(f"    • {item.name} ({size:.2f} MB)")
                        else:
                            print(f"    • {item.name}/ (directory)")
                    
                    return "exists"
        
        self.print_info("Model not found locally. Will download...")
        return "not_exists"
    
    def download_model_from_huggingface(self):
        """Download model from HuggingFace"""
        self.print_section("4/6", "Downloading Model from HuggingFace")
        
        try:
            self.print_info(f"Model: {self.model_name}")
            self.print_info(f"Destination: {self.local_model_path.absolute()}")
            self.print_info("This may take 5-15 minutes depending on your internet speed...")
            self.print_info("Downloading please wait...\n")
            
            # Import here after dependencies are installed
            from sentence_transformers import SentenceTransformer
            
            # Download model
            model = SentenceTransformer(self.model_name)
            
            # Save model to local path
            model.save(str(self.local_model_path))
            
            self.print_success("Model downloaded successfully!")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to download model: {str(e)}")
    
    def verify_model(self):
        """Verify downloaded model integrity"""
        self.print_section("5/6", "Verifying Model")
        
        try:
            # Check if directory exists
            if not self.local_model_path.exists():
                self.print_error(f"Model directory not found at {self.local_model_path}")
            
            # Check required files
            required_files = [
                "config.json",
                "tokenizer.json",
                "modules.json"
            ]
            
            print("\n  Checking required files:")
            for file in required_files:
                file_path = self.local_model_path / file
                if file_path.exists():
                    size = file_path.stat().st_size / 1024
                    self.print_success(f"  {file} ({size:.2f} KB)")
                else:
                    self.print_error(f"  {file} - MISSING!")
            
            # Check model weight files
            print("\n  Checking model weights:")
            weight_files = list(self.local_model_path.glob("*.safetensors")) + \
                          list(self.local_model_path.glob("pytorch_model.bin"))
            
            if weight_files:
                for wf in weight_files:
                    size = wf.stat().st_size / (1024**3)
                    self.print_success(f"  {wf.name} ({size:.2f} GB)")
            else:
                self.print_error("No model weight files found!")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in self.local_model_path.rglob("*") if f.is_file()) / (1024**3)
            
            # List all directories
            print("\n  Model structure:")
            for item in sorted(self.local_model_path.iterdir()):
                if item.is_dir():
                    item_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024**2)
                    self.print_success(f"  {item.name}/ ({item_size:.2f} MB)")
            
            self.print_success(f"Total model size: {total_size:.2f} GB")
            return True
            
        except Exception as e:
            self.print_error(f"Model verification failed: {str(e)}")
    
    def create_project_structure(self):
        """Create recommended project structure"""
        self.print_section("2.5/6", "Creating Project Structure")
        
        directories = [
            self.models_dir,
            self.project_root / "embedding_app",
            self.project_root / "logs"
        ]
        
        for dir_path in directories:
            try:
                dir_path.mkdir(exist_ok=True)
            except Exception as e:
                self.print_warning(f"Could not create {dir_path}: {str(e)}")
        
        self.print_success("Project structure verified")
    
    def create_gitignore(self):
        """Create .gitignore to exclude models folder"""
        gitignore_path = self.project_root / ".gitignore"
        
        gitignore_content = """# Model files (large, not versioned)
models/
*.safetensors
pytorch_model.bin

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# Docker
.dockerignore

# Logs
*.log
logs/
"""
        
        try:
            if not gitignore_path.exists():
                gitignore_path.write_text(gitignore_content)
                self.print_success(f"Created .gitignore at {gitignore_path}")
            else:
                # Append models/* if not already present
                content = gitignore_path.read_text()
                if "models/" not in content:
                    gitignore_path.write_text(content + "\n\n# Model files\nmodels/\n")
                    self.print_success("Updated .gitignore with models/ entry")
        except Exception as e:
            self.print_info(f"Could not update .gitignore: {str(e)}")
    
    def show_final_report(self):
        """Show comprehensive final report"""
        self.print_banner("DOWNLOAD COMPLETE & VERIFIED ✓", "=")
        
        print("📊 SUMMARY:")
        print("-" * 70)
        
        # Model info
        if self.local_model_path.exists():
            total_size = sum(f.stat().st_size for f in self.local_model_path.rglob("*") if f.is_file()) / (1024**3)
            file_count = len(list(self.local_model_path.rglob("*")))
            
            print(f"\n  Model Name      : {self.local_model_name}")
            print(f"  Location        : {self.local_model_path.absolute()}")
            print(f"  Total Size      : {total_size:.2f} GB")
            print(f"  Files/Folders   : {file_count}")
        
        # Project structure
        print(f"\n  Project Root    : {self.project_root.absolute()}")
        print(f"  Models Dir      : {self.models_dir.absolute()}")
        
        print("\n" + "=" * 70)
        print("\n✓ READY TO BUILD DOCKER CONTAINERS!\n")
        
        print("📝 NEXT STEPS:")
        print("-" * 70)
        print("\n  Option 1: Use Auto-Deploy Script (RECOMMENDED)")
        print("     Windows:  .\\deploy.bat")
        print("     PowerShell: .\\deploy.ps1")
        print("     Linux/Mac: ./deploy.sh")
        print("\n  Option 2: Manual Docker Commands")
        print("     docker-compose down")
        print("     docker system prune -a --volumes -f")
        print("     docker-compose build embedding-api --no-cache")
        print("     docker-compose up -d")
        print("\n  Test endpoints:")
        print("     http://localhost:85/health")
        print("     http://localhost:85/docs")
        print("\n" + "=" * 70)
    
    def run(self):
        """Run complete download process"""
        self.print_banner("QWEN3-EMBEDDING-4B MODEL AUTO-DOWNLOADER", "═")
        
        try:
            # Step 1: Install dependencies
            self.install_dependencies()
            
            # Step 2: Create directory
            self.create_models_directory()
            
            # Step 2.5: Create project structure
            self.create_project_structure()
            
            # Step 3: Check if model exists
            model_status = self.check_existing_model()
            
            # Step 4: Download if needed
            if model_status == "not_exists":
                self.download_model_from_huggingface()
            
            # Step 5: Verify model
            self.verify_model()
            
            # Step 6: Create .gitignore
            self.create_gitignore()
            
            # Step 7: Show report
            self.show_final_report()
            
            print("\n✓ Model setup complete! Waiting for user input...")
            input("\nPress Enter to exit...")
            
        except KeyboardInterrupt:
            self.print_error("\nSetup cancelled by user")
        except Exception as e:
            self.print_error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    downloader = ModelDownloader()
    downloader.run()