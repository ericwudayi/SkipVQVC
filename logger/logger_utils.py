import os

def prepare_directories_and_logger(logger_class, output_directory = 'output', 
	log_directory = 'log'):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logger = logger_class(os.path.join(output_directory, log_directory))

    return logger