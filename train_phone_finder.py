from utils.tools import parse_command_line_args, check_folder_contents


def main():
    args = parse_command_line_args()

    data_folder_path = args.data_folder

    if check_folder_contents(data_folder_path):
        print('Verification successful. You can proceed with processing.')

    else:
        print('Please provide a valid data_folder that meets the required format for training.')


if __name__ == "__main__":
    main()
