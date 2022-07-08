def read_cuda_file(cuda_path):
    f = open(cuda_path, 'r')
    source_line = ""
    while True:
        line = f.readline()
        if not line: break
        source_line = source_line + line
    f.close()
    return source_line