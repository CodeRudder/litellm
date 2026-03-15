#!/bin/bash

docker logs litellm-verify 2>&1 | grep -oP "attempted_retries': [1-9]"

