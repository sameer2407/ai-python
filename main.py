# main.py
import grpc
from concurrent import futures
from generated import pdf_service_pb2  # Changed from rag_pb2
from generated import pdf_service_pb2_grpc  # Changed from rag_pb2_grpc
from rag_engine import RAGEngine
import os
from dotenv import load_dotenv

load_dotenv()

class PDFServicer(pdf_service_pb2_grpc.PDFServiceServicer):
    def __init__(self):
        self.rag_engine = RAGEngine(
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.current_document = None

    def ProcessPDF(self, request, context):
        try:
            # Save PDF temporarily
            temp_path = f"/tmp/{request.filename}"
            with open(temp_path, "wb") as f:
                f.write(request.pdf_content)
            
            # Only process if it's a new document
            if self.current_document != request.filename:
                print(f"Processing new document: {request.filename}")
                # Process with RAG engine
                documents = self.rag_engine.load_documents(temp_path)
                self.rag_engine.process_documents(documents)
                self.current_document = request.filename
                print(f"Document processed successfully: {request.filename}")
            else:
                print(f"Document already processed: {request.filename}")
            
            # Cleanup
            os.remove(temp_path)
            
            return pdf_service_pb2.ProcessPDFResponse(
                success=True,
                error_message=""
            )
        except Exception as e:
            return pdf_service_pb2.ProcessPDFResponse(
                success=False,
                error_message=str(e)
            )

    def QueryPDF(self, request, context):
        try:
            response = self.rag_engine.query(request.query)
            return pdf_service_pb2.QueryPDFResponse(
                answer=response,
                success=True,
                error_message=""
            )
        except Exception as e:
            return pdf_service_pb2.QueryPDFResponse(
                success=False,
                error_message=str(e)
            )

    def StreamQueryPDF(self, request, context):
        """
        Stream the response for a query in real-time.
        """
        try:
            for chunk in self.rag_engine.stream_query(request.query):
                yield pdf_service_pb2.StreamQueryPDFResponse(
                    chunk=chunk,
                    success=True,
                    error_message=""
                )
        except Exception as e:
            yield pdf_service_pb2.StreamQueryPDFResponse(
                chunk="",
                success=False,
                error_message=str(e)
            )

def serve():
    # Get port from environment variable
    port = os.getenv('PORT', '50051')

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )

    pdf_service_pb2_grpc.add_PDFServiceServicer_to_server(
        PDFServicer(), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()