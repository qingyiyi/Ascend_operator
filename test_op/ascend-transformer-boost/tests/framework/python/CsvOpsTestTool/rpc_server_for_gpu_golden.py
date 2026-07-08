#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import pickle
from multiprocessing.connection import Listener
from threading import Thread
import torch
import json
import argparse

class RPCHandler:
    gpu_golden_functions = {}

    @staticmethod
    def register_gpu_golden(op_name:str):
        RPCHandler.gpu_golden_functions[op_name] = op_name + '.gpu_golden'

    @staticmethod
    def handle_connection(connection):
        try:
            while True:
                # Receive a message
                op_name, args, kwargs = pickle.loads(connection.recv())
                # Run the RPC and send a response
                try:
                    result = eval(RPCHandler.gpu_golden_functions[op_name])(*args,**kwargs)
                    connection.send(pickle.dumps(result))
                except Exception as e:
                    connection.send(pickle.dumps(e))
        except EOFError:
             pass

# Run the server
def rpc_server(handler, address):
    sock = Listener(address)
    while True:
        client = sock.accept()
        t = Thread(target=handler.handle_connection, args=(client,))
        t.daemon = True
        t.start()

class ElewiseOperation:
    ELEWISE_ADD = 8
    def gpu_golden(intensors, op_params):
        json_data = json.loads(op_params)
        elewiseType = json_data["elewiseType"]
        if elewiseType == ElewiseOperation.ELEWISE_ADD:
            print("intensors[0]: ", intensors[0])
            print("intensors[1]: ", intensors[1])
            gpu_golden = intensors[0] + intensors[1]
            print("gpu_golden: ", gpu_golden)
            return [gpu_golden]

# Register with a handler
RPCHandler.register_gpu_golden('ElewiseOperation')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gpu golden executor")
    parser.add_argument('-ip', default='', help='gpu server\'s ip')
    parser.add_argument('-port', default=0, type=int, help='gpu server\'s port')
    args = parser.parse_args()
    if args.ip == '' or args.port == 0:
        print("ip and port must be provided!")
        exit(1)
    rpc_server(RPCHandler, (args.ip, args.port))
