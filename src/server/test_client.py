# coding: utf-8
from __future__ import print_function

import grpc

import fn_pb2
import fn_pb2_grpc

def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = fn_pb2_grpc.NNStub(channel)
  response = stub.GetSegmentation(fn_pb2.Request(
    img_filepath='/opt/example/input/Mapillary/Mapillary__validation/img/-3-MmXdwhyIQhtb4-8NqHQ.jpg'
  ))
  print('Done, responce: {}'.format(response))

if __name__ == '__main__':
  run()