import os
import gin

def gin_register_and_parse(gin_config_file: str):
    gin.add_config_file_search_path(os.path.join(os.getcwd(), 'gin'))
    gin.add_config_file_search_path(os.path.join(os.path.dirname(os.getcwd()), 'gin'))
    gin.parse_config_file(gin_config_file)