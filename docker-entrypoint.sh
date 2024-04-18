#!/bin/bash
set -e

# if [[ "$1" = "serve" ]]; then
shift 1
torchserve --start --ts-config /home/model-server/config.properties \
    --model-store /home/model-server/model-store/ \
    --models fer.mar
    # sleep 5
    # curl -X POST "http://0.0.0.0:8081/workflows?url=as_wf.war"
# else
#     eval "$@"
# fi

# prevent docker exit
tail -f /dev/null