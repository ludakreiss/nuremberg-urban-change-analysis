from pathlib import Path
import subprocess
import sys


class PipelineWorker:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "output"
        self.scripts_dir = self.project_root / "scripts"
        self.src_dir = self.project_root / "src"

        self.master_dataset_path = self.output_dir / "nuremberg_dataset_final.csv"
        self.features_labels_path = (
            self.data_dir / "labels" / "combined_format" / "nuremberg_features_labels.parquet"
        )
        self.model_results_path = (
            self.output_dir / "modeling_results" / "all_tasks_results.csv"
        )
        self.dashboard_path = self.src_dir / "ui" / "dashboard.py"

    def _run_python_file(self, script_path: Path):
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        print(f"\n[RUN] {script_path.name}")
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            check=True,
        )

    def _ensure_exists(self, path: Path, label: str):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        print(f"[OK] {label}: {path}")

    def download_data(self):
        print("\nDownload HF data")
        script = self.scripts_dir / "download_hf_data.py"
        self._run_python_file(script)

    def build_master_dataset(self):
        print("\nBuild master dataset")
        script = self.scripts_dir / "build_master_dataset.py"
        self._run_python_file(script)
        self._ensure_exists(self.master_dataset_path, "Master dataset")

    def run_feature_engineering(self):
        print("\nFeature engineering")
        script = self.scripts_dir / "feature_engineering.py"
        self._run_python_file(script)
        self._ensure_exists(self.features_labels_path, "Feature-label parquet")

    def run_modeling(self):
        print("\nModeling")
        script = self.scripts_dir / "run_modeling.py"
        self._run_python_file(script)
        self._ensure_exists(self.model_results_path, "Model results")

    def launch_ui(self):
        print("\nLaunch UI")
        self._ensure_exists(self.dashboard_path, "Dashboard file")

        subprocess.run(
            ["streamlit", "run", str(self.dashboard_path)],
            cwd=str(self.project_root),
            check=True,
        )