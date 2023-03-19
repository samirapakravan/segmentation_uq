# segment_uq
## Image Segmentation with Uncertainty Quantification
Following the paper https://pubmed.ncbi.nlm.nih.gov/31729078/ 

## Without Docker
TBD ...

## With Docker
Create an account on ``ngc.nvidia.com`` and ``Create an API key`` and on your machine do ``docker login nvcr.io``, use the provided user/password to login to the ngc server.

To build the docker image ``./launch.sh build`` [ONLY the first time!] and to start it do ``./launch.sh dev -d``. Then in VS Code `` Ctrl+Shift+P`` and choose ``Attach to Running Container`` and choose ``vision`` from the running containers list.
