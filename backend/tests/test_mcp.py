import asyncio
import json

from app.mcp_server import server


def _call(name, args):
    result = asyncio.run(server.call_tool(name, args))
    return result


def test_mcp_exposes_only_read_only_tools():
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "list_documents",
        "get_document",
        "find_documents_by_vehicle",
        "search_contract",
        "fleet_alerts",
        "check_identifiers",
    }


def test_mcp_tools_run():
    out = json.dumps(_call("check_identifiers", {"gstin": "27AAACR5055K1Z7", "vehicle_no": "TS 09 UB 4521"}), default=str)
    assert "true" in out.lower()
    hits = json.dumps(_call("search_contract", {"query": "detention"}), default=str)
    assert "2,500" in hits
