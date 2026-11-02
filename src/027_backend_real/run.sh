#!/bin/bash
export PATH="/app/.venv/bin:$PATH"
python -m backend_real.nodes.broker &
python -m backend_real.nodes.camera &
python -m backend_real.nodes.vision &
python -m backend_real.nodes.actuator &
python -m backend_real.nodes.logger &
wait -n
exit $?
