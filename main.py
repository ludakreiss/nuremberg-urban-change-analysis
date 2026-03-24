from orchestrator import PipelineOrchestrator


def main():
    orchestrator = PipelineOrchestrator(
        run_download=True,
        run_build_master=True,
        run_feature_engineering=True,
        run_modeling=True,
        launch_ui=True,   
    )
    orchestrator.run()


if __name__ == "__main__":
    main()