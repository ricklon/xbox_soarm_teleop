"""soarm-hub --preset sorting: the student sorting session in one option.

Trying the sorting work took five options (scene, no coordinates, pointer,
judge) and still could not rename objects on the hub. A preset sets them all;
anything given explicitly still wins. --names reaches every arm.
"""

import logging
import warnings

from xbox_soarm_teleop.cli.hub import PRESETS, apply_preset, build_parser

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


def _args(*argv):
    return apply_preset(build_parser().parse_args(list(argv)))


def test_the_sorting_preset_sets_up_the_sorting_session():
    args = _args("--preset", "sorting")
    assert args.scene == "sort_colors"
    assert args.no_coordinates is True
    assert args.pointer == "openrouter:google/gemma-4-31b-it"
    assert args.judge == "clef"
    assert args.names == "truthful"
    assert "sorting" in PRESETS


def test_explicit_options_beat_the_preset():
    args = _args("--preset", "sorting", "--pointer", "oracle", "--judge", "none", "--names", "neutral")
    assert args.pointer == "oracle" and args.judge == "none" and args.names == "neutral"
    assert args.scene == "sort_colors"


def test_no_preset_changes_nothing():
    args = _args()
    assert args.scene == "pick_place_easy" and args.pointer is None and args.judge is None


def test_names_reach_every_arm():
    from xbox_soarm_teleop.twin.hub import Hub

    hub = Hub(seats=2, scene="sort_colors", log_dir=None, naming="neutral")
    try:
        names = [room.service.shown_names for room in hub.rooms.values()]
        assert all(
            n is not None and set(n.values()) == {"object_1", "object_2", "object_3"} for n in names
        )
    finally:
        hub.close()


def test_the_page_suggests_prompts_for_the_scene_it_shows():
    from starlette.testclient import TestClient

    from xbox_soarm_teleop.twin.hub import Hub
    from xbox_soarm_teleop.twin.hub_server import STARTER_PROMPTS, build_hub_app

    hub = Hub(seats=1, scene="sort_colors", log_dir=None)
    try:
        token = next(iter(hub.rooms))
        client = TestClient(build_hub_app(hub), base_url="http://127.0.0.1:8800")
        prompts = client.get(f"/r/{token}/prompts").json()["prompts"]
        text = " ".join(p["text"] for p in prompts)
        assert prompts != STARTER_PROMPTS
        assert "bean bag" not in text and "goal box" not in text
        assert "pick" in text and "check_success" in text and "by colour" in text
        assert "sort_colors_decoy" in text  # the ambiguous one
    finally:
        hub.close()


def test_the_page_says_where_each_object_is_even_when_agents_cannot():
    from xbox_soarm_teleop.twin import scenarios
    from xbox_soarm_teleop.twin.hub import Hub
    from xbox_soarm_teleop.twin.hub_server import room_summary

    hub = Hub(seats=1, scene="sort_colors", log_dir=None, hide_positions=True, naming="neutral")
    try:
        room = next(iter(hub.rooms.values()))
        service = room.service
        scenarios.place_in_bin(service, "pick_cube", "blue_bin")
        for _ in range(30):
            service.tick()
        summary = room_summary(room)
        shown = service.shown_names
        assert summary["where"][shown["pick_cube"]] == "in the blue bin"
        assert summary["where"][shown["green_cube"]] == "on the table"
        assert "xyz_m" not in str(summary["where"])
        assert "pick_cube" not in summary["where"]  # the names the student's agent sees
    finally:
        hub.close()
