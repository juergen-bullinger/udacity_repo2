#!/bin/bash

autopep8 --in-place --aggressive --aggressive $@
pylint $@
