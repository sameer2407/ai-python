import grpc
from generated import pdf_service_pb2
from generated import pdf_service_pb2_grpc
import os

def run():
    # Create a channel to the server
    with grpc.insecure_channel('localhost:50051') as channel:
        # Create a stub (client)
        stub = pdf_service_pb2_grpc.PDFServiceStub(channel)
        
        # Test PDF processing
        try:
            # Read a PDF file
            pdf_path = "Metamorphosis.pdf"  # PDF in current directory
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            
            print(f"Processing PDF: {pdf_path}")
            # Process the PDF
            process_response = stub.ProcessPDF(
                pdf_service_pb2.ProcessPDFRequest(
                    pdf_content=pdf_content,
                    filename=os.path.basename(pdf_path)
                )
            )
            
            if process_response.success:
                print("PDF processed successfully!")
                
                # Test queries
                queries = [
                    "What is the main topic of this document?",
                    "Summarize the key points",
                    "What are the main findings?"
                ]
                
                for query in queries:
                    print(f"\nSending query: {query}")
                    try:
                        query_response = stub.QueryPDF(
                            pdf_service_pb2.QueryPDFRequest(
                                query=query
                            )
                        )
                        
                        if query_response.success:
                            print(f"Answer: {query_response.answer}")
                        else:
                            print(f"Error: {query_response.error_message}")
                    except Exception as e:
                        print(f"Error processing query: {str(e)}")
            else:
                print(f"Error processing PDF: {process_response.error_message}")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == '__main__':
    run()
