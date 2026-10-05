from concurrent import futures

import grpc

from .generated import messages_pb2 as pb
from .generated import messages_pb2_grpc as pb_grpc


class Greeter(pb_grpc.GreeterServiceServicer):
    def SayHello(self, request, context):
        return pb.SayHelloResponse(message=f"Hello {request.name} from PYTHON!")


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_GreeterServiceServicer_to_server(Greeter(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


def main():
    serve()


if __name__ == "__main__":
    main()
