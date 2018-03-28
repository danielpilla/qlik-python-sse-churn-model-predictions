#! /usr/bin/env python3
import argparse
import json
import logging
import logging.config
import os
import sys
import time
from concurrent import futures
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import ServerSideExtension_pb2 as SSE
import grpc
from SSEData_churn import FunctionType, \
                               get_func_type
from ScriptEval_churn import ScriptEval

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# LOAD THE SCALER
scaler = joblib.load('Scaler/scaler.sav') 

class ExtensionService(SSE.ConnectorServicer):
    """
    A simple SSE-plugin created for the ARIMA example.
    """

    def __init__(self, funcdef_file):
        """
        Class initializer.
        :param funcdef_file: a function definition JSON file
        """
        self._function_definitions = funcdef_file
        self.scriptEval = ScriptEval()
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.config.fileConfig('logger.config')
        logging.info('Logging enabled')

    @property
    def function_definitions(self):
        """
        :return: json file with function definitions
        """
        return self._function_definitions

    @property
    def functions(self):
        """
        :return: Mapping of function id and implementation
        """
        return {
            0: '_churn'
        }

    @staticmethod
    def _get_function_id(context):
        """
        Retrieve function id from header.
        :param context: context
        :return: function id
        """
        metadata = dict(context.invocation_metadata())
        header = SSE.FunctionRequestHeader()
        header.ParseFromString(metadata['qlik-functionrequestheader-bin'])

        return header.functionId

    """
    Implementation of added functions.
    """

    @staticmethod
    def _churn(request, context):
        # Disable caching.
        md = (('qlik-cache', 'no-store'),)
        context.send_initial_metadata(md)

        # instantiate list
        modelName = None
        columnNames = None
        dataList = []

        for request_rows in request:

            for row in request_rows.rows:
                # model name - only grab once as it is repeated
                if not modelName:
                    modelName = ''
                    modelName = 'Models/' + [d.strData for d in row.duals][0] + '.pkl'

                # column names - only grab once as it is repeated
                if not columnNames:
                    columnNames = ''
                    columnNames = [d.strData.replace('\\',' ').replace('[','').replace(']','') for d in row.duals][1]
                    columnNames = columnNames.split('|')
                
                # pull duals from each row, and the strData from duals
                data = [d.strData.split('|') for d in row.duals][2]
                dataList.append(data)

        churn_df = pd.DataFrame(dataList)
        churn_df.columns = columnNames

        # ARBITRARY FUNCTION TO CLEANSE DATA
        def cleanse(df):
            global scaler
            
            # Isolate target data
            churn_result = df['Churn?']
            y = np.where(churn_result == 'True.',1,0)

            # We don't need these columns
            to_drop = ['State','Area Code','Phone','Churn?']
            churn_feat_space = df.drop(to_drop,axis=1)

            # 'yes'/'no' has to be converted to boolean values
            # NumPy converts these from boolean to 1. and 0. later
            yes_no_cols = ["Int'l Plan","VMail Plan"]
            churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

            X = churn_feat_space.as_matrix().astype(np.float)

##            scaler = StandardScaler()
            X = scaler.transform(X)

            return X

        # LOAD THE MODEL
        clf = joblib.load(modelName)

        # PREDICT
        df = pd.DataFrame(cleanse(churn_df))
##        df = cleanse(pd.read_csv("ChurnPredictData.csv"))
        predictions = clf.predict(df)
        
        # Create an iterable of dual with the result
        duals = iter([[SSE.Dual(numData=d)] for d in predictions])

        # Yield the row data as bundled rows
        yield SSE.BundledRows(rows=[SSE.Row(duals=d) for d in duals])


    """
    Implementation of rpc functions.
    """

    def GetCapabilities(self, request, context):
        """
        Get capabilities.
        Note that either request or context is used in the implementation of this method, but still added as
        parameters. The reason is that gRPC always sends both when making a function call and therefore we must include
        them to avoid error messages regarding too many parameters provided from the client.
        :param request: the request, not used in this method.
        :param context: the context, not used in this method.
        :return: the capabilities.
        """
        logging.info('GetCapabilities')
        # Create an instance of the Capabilities grpc message
        # Enable(or disable) script evaluation
        # Set values for pluginIdentifier and pluginVersion
        capabilities = SSE.Capabilities(allowScript=True,
                                        pluginIdentifier='Hello World - Qlik',
                                        pluginVersion='v1.0.0-beta1')

        # If user defined functions supported, add the definitions to the message
        with open(self.function_definitions) as json_file:
            # Iterate over each function definition and add data to the capabilities grpc message
            for definition in json.load(json_file)['Functions']:
                function = capabilities.functions.add()
                function.name = definition['Name']
                function.functionId = definition['Id']
                function.functionType = definition['Type']
                function.returnType = definition['ReturnType']

                # Retrieve name and type of each parameter
                for param_name, param_type in sorted(definition['Params'].items()):
                    function.params.add(name=param_name, dataType=param_type)

                logging.info('Adding to capabilities: {}({})'.format(function.name,
                                                                     [p.name for p in function.params]))

        return capabilities

    def ExecuteFunction(self, request_iterator, context):
        """
        Execute function call.
        :param request_iterator: an iterable sequence of Row.
        :param context: the context.
        :return: an iterable sequence of Row.
        """
        # Retrieve function id
        func_id = self._get_function_id(context)

        # Call corresponding function
        logging.info('ExecuteFunction (functionId: {})'.format(func_id))

        return getattr(self, self.functions[func_id])(request_iterator, context)

    def EvaluateScript(self, request, context):
        """
        This plugin provides functionality only for script calls with no parameters and tensor script calls.
        :param request:
        :param context:
        :return:
        """
        # Parse header for script request
        metadata = dict(context.invocation_metadata())
        header = SSE.ScriptRequestHeader()
        header.ParseFromString(metadata['qlik-scriptrequestheader-bin'])

        # Retrieve function type
        func_type = get_func_type(header)

        # Verify function type
        if (func_type == FunctionType.Aggregation) or (func_type == FunctionType.Tensor):
            return self.scriptEval.EvaluateScript(header, request, func_type)
        else:
            # This plugin does not support other function types than aggregation  and tensor.
            raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED,
                                'Function type {} is not supported in this plugin.'.format(func_type.name))

    """
    Implementation of the Server connecting to gRPC.
    """

    def Serve(self, port, pem_dir):
        """
        Sets up the gRPC Server with insecure connection on port
        :param port: port to listen on.
        :param pem_dir: Directory including certificates
        :return: None
        """
        # Create gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        SSE.add_ConnectorServicer_to_server(self, server)

        if pem_dir:
            # Secure connection
            with open(os.path.join(pem_dir, 'sse_server_key.pem'), 'rb') as f:
                private_key = f.read()
            with open(os.path.join(pem_dir, 'sse_server_cert.pem'), 'rb') as f:
                cert_chain = f.read()
            with open(os.path.join(pem_dir, 'root_cert.pem'), 'rb') as f:
                root_cert = f.read()
            credentials = grpc.ssl_server_credentials([(private_key, cert_chain)], root_cert, True)
            server.add_secure_port('[::]:{}'.format(port), credentials)
            logging.info('*** Running server in secure mode on port: {} ***'.format(port))
        else:
            # Insecure connection
            server.add_insecure_port('[::]:{}'.format(port))
            logging.info('*** Running server in insecure mode on port: {} ***'.format(port))

        # Start gRPC server
        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', nargs='?', default='50072')
    parser.add_argument('--pem_dir', nargs='?')
    parser.add_argument('--definition-file', nargs='?', default='FuncDefs_churn.json')
    args = parser.parse_args()

    # need to locate the file when script is called from outside it's location dir.
    def_file = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.definition_file)

    calc = ExtensionService(def_file)
    calc.Serve(args.port, args.pem_dir)
