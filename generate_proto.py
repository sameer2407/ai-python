import os
import subprocess
import grpc_tools.protoc

def generate_proto():
    proto_file = "proto/pdf_service.proto"
    output_dir = "."
    
    # Create the command to generate Python code from proto file
    command = [
        "python", "-m", "grpc_tools.protoc",
        f"--proto_path={os.path.dirname(proto_file)}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file
    ]
    
    # Run the command
    subprocess.run(command, check=True)
    print("Generated Python code from proto file successfully!")

def generate_python_proto():
    # Create the generated directory
    generated_dir = 'generated'
    os.makedirs(generated_dir, exist_ok=True)
    
    # Create __init__.py to make it a package
    with open(os.path.join(generated_dir, '__init__.py'), 'w') as f:
        f.write('')
    
    # Generate the proto files
    grpc_tools.protoc.main([
        'grpc_tools.protoc',
        '-I./proto',
        f'--python_out={generated_dir}',
        f'--grpc_python_out={generated_dir}',
        './proto/pdf_service.proto'
    ])
    
    # Modify the import in pdf_service_pb2_grpc.py
    grpc_file = os.path.join(generated_dir, 'pdf_service_pb2_grpc.py')
    with open(grpc_file, 'r') as f:
        content = f.read()
    
    # Replace the import statement
    content = content.replace(
        'import pdf_service_pb2 as pdf__service__pb2',
        'from . import pdf_service_pb2 as pdf__service__pb2'
    )
    
    with open(grpc_file, 'w') as f:
        f.write(content)
    
    print(f"Generated Python code in {generated_dir}")

if __name__ == '__main__':
    generate_python_proto()