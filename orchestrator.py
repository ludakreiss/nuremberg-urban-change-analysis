from pathlib import Path
from pipeline_worker import PipelineWorker


class PipelineOrchestrator:
    def __init__(
        self,
        run_download=True,
        run_build_master=True,
        run_feature_engineering=True,
        run_modeling=True,
        launch_ui=False,
    ):
        self.project_root = Path(__file__).resolve().parent
        self.worker = PipelineWorker(self.project_root)

        self.run_download = run_download
        self.run_build_master = run_build_master
        self.run_feature_engineering = run_feature_engineering
        self.run_modeling = run_modeling
        self.launch_ui = launch_ui

    def run(self):
        print("\n" + "=" * 90)
        print("NUREMBERG URBAN CHANGE ANALYSIS PIPELINE")
        print("=" * 90)

        if self.run_download:
            self.worker.download_data()

        if self.run_build_master:
            self.worker.build_master_dataset()

        if self.run_feature_engineering:
            self.worker.run_feature_engineering()

        if self.run_modeling:
            self.worker.run_modeling()

        if self.launch_ui:
            self.worker.launch_ui()

        print("\n" + "=" * 90)
        print("PIPELINE FINISHED SUCCESSFULLY")
        print("=" * 90)