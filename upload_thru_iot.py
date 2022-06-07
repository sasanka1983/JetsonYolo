#!/usr/bin/env python3

from azure.iot.device import IoTHubDeviceClient
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobClient
import os
import sys


class Upload_thru_iot:

    def __init__(self,connectionString):
        self.__iot_client = IoTHubDeviceClient.create_from_connection_string(connectionString)   
    
    def __del__(self):
        self.__iot_client.shutdown()

    def __upload_blob(self,azure_blob_details,file_name):
        try:
            url="https://{host}/{container}/{blob}{token}".format(host=azure_blob_details["hostName"],container=azure_blob_details["containerName"],blob=azure_blob_details["blobName"],token=azure_blob_details["sasToken"])

            with BlobClient.from_blob_url(url) as blob_client:
                with open(file_name,"rb") as f:
                    result=blob_client.upload_blob(f,overwrite=True)
                    return(True, result)

        except FileNotFoundError as fex:
            fex.status_code =404
            return (False, fex)

        except AzureError as aex:
            return (False,aex)

    def upload_image(self,filePath):
        self.__iot_client.connect()
        blob_name=os.path.basename(filePath)
        storage_info=self.__iot_client.get_storage_info_for_blob(blob_name)

        success,result = self.__upload_blob(storage_info,filePath)

        if success ==True:
            print("upload successful")
            print(result)
            self.__iot_client.notify_blob_upload_status(storage_info["correlationId"],True,200,"OK:{}".format(filePath))
        else:
            print("upload failed. Error is: ")
            print (result)
            self.__iot_client.notify_blob_upload_status(storage_info["correlationId"],False,result.status_code,str(result).format(filePath))

def main(args):
    thisObj=Upload_thru_iot(args[0])
    try:
        print("upload a file to IoT Hub")
        thisObj.upload_image(args[1])
    except KeyboardInterrupt:
        print("IoT Hub client stopped")

if __name__ == '__main__':
    main(sys.argv)
        