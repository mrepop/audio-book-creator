#!/bin/bash
# Dev server launcher - starts both backend and frontend
# Ctrl+C cleanly stops both processes

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Kill anything on our ports first
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
sleep 0.5

# Trap Ctrl+C to kill both child processes
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# Start backend
echo "Starting backend on :8000..."
PYTHONPATH="$PROJECT_ROOT" "$PROJECT_ROOT/venv/bin/python" -m src.api.main &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on :5173..."
npm --prefix "$PROJECT_ROOT/frontend" run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  Press Ctrl+C to stop both"
echo ""

# Wait for either to exit
wait $BACKEND_PID $FRONTEND_PID
