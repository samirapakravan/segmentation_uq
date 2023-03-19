# segment

# Dev
Create an account on ``ngc.nvidia.com`` and ``Create an API key`` and on your machine do ``docker login nvcr.io``, use the provided user/password to login to the ngc server.

To build the docker image ``./launch.sh build`` [ONLY the first time!] and to start it do ``./launch.sh dev -d``. Then in VS Code `` Ctrl+Shift+P`` and choose ``Attach to Running Container`` and choose ``pkpd`` from the running containers list.

`process_data.py` -> `data_split.py` -> `generate_interp_nodosing.py` -> `data_parse.py`