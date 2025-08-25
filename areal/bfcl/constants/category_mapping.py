from pathlib import Path

VERSION_PREFIX = "BFCL_v3"

PACKAGE_ROOT = Path(__file__).parent.parent.parent  # TODO 确认项目的路径

PROMPT_PATH = PACKAGE_ROOT / "bfcl/data"
MULTI_TURN_FUNC_DOC_PATH = PROMPT_PATH / "multi_turn_func_doc"
POSSIBLE_ANSWER_PATH = PROMPT_PATH / "possible_answer"

TEST_FILE_MAPPING = {
    "multi_turn_base": f"{VERSION_PREFIX}_multi_turn_base.json",
}

TEST_COLLECTION_MAPPING = {
    "multi_turn": [
        "multi_turn_base",
    ]
}

MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}