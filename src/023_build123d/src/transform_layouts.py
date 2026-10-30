import glob, json

W, L, O = 0.290, 0.245, 0.0075
fx = lambda x: x - W / 2 + O
fy = lambda y: y - L / 2 + O

for p in glob.glob("src/labyrinth_game_cad/layouts/cyberrunner_*_layout.json"):
    with open(p) as f:
        d = json.load(f)
    d["holes"] = [[fx(x), fy(y)] for x, y in d.get("holes", [])]
    d["waypoints"] = [[fx(x), fy(y)] for x, y in d.get("waypoints", [])]
    d["walls_v"] = [[fy(a), fy(b), fx(x)] for a, b, x in d.get("walls_v", [])]
    d["walls_h"] = [[fx(a), fx(b), fy(y)] for a, b, y in d.get("walls_h", [])]
    with open(p, "w") as f:
        json.dump(d, f, indent=2)
