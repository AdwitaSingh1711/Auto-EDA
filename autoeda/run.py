import argparse

from .AutoEDA import AutoEDA

def main():
    """Main function for CLI"""
    
    parser = argparse.ArgumentParser(
        description="Perform automatic EDA for your CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--file_type", type=str, required=True, help="File extension type(eg 'csv')")
    parser.add_argument("--use_llm", action="store_true", help="Use local LM to provide EDA suggestions")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations for the data")
    parser.add_argument("--save_report", action="store_true", help="Save the EDA output to file")
    parser.add_argument("--output_format", choices=["txt","json"], default="txt", help="Output report format")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model to be used for analysis")
    parser.add_argument("--output_path", type=str, default = "eda_report", help="path of directory where report is to be saved")
    parser.add_argument("--plot_dir", type=str, default = "eda_plots", help="Directory to save visualisation plots")

    args=parser.parse_args()

    try:
        if not args.file_path or not args.file_type:
            raise ValueError("You either forgot to specify the file type or the file path.")

        # INITIALISE AND RUN AUTO-EDA
        auto_eda = AutoEDA(model_name = args.model, plot_output_dir = args.plot_dir, output_path = args.output_path)

        results = auto_eda.run_eda(
                    file_path = args.file_path,
                    file_type = args.file_type,
                    use_llm = args.use_llm,
                    visualize = args.visualize,
                    save_report = args.save_report,
                    output_format = args.output_format,
                    output_path = args.output_path,
                    plot_output_dir = args.plot_dir
                )

        auto_eda.print_basic_results(results, args.output_path)

    except Exception as e:
        logger.error("You did something wrong somewhere my bro:{e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
