const tf = require('@tensorflow/tfjs');

mapLoader = require('./data');

module.exports =
    class Game {
        constructor() {
            this.dir = 0;
            this.dirs = [
                [1, 0],
                [1, 1],
                [0, 1],
                [-1, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [1, -1],
            ];
            this.dirsA = []
            for (let i = 0; i < 8; i++) this.dirsA.push(Math.atan2(this.dirs[i][1], this.dirs[i][0]));
            this.map = [];
            this.rows = 0;
            this.cols = 0;
            this.pos = [0, 0];
        }
        init() {
            return new Promise((resolve, reject) => {
                mapLoader.import("maps/floor_plan.png").then((data) => {
                    this.rows = data.shape[1];
                    this.cols = data.shape[0];
                    for (let y = 0; y < this.cols; y++) {
                        this.map.push(data.map.slice(y * this.rows, (y + 1) * this.rows))
                    }
                    this.shape = data.shape;
                    this.pos[0] = this.rows / 2;
                    this.pos[1] = this.cols / 2;
                    resolve();
                }).catch((msg) => reject(msg));
            });
        }
        ray(x0, y0, x1, y1, color = 0) {
            //console.log(x0, y0, x1, y1)
            let dx = x1 - x0;
            let dy = y1 - y0;
            let dys = Math.sign(dy);
            let dxs = Math.sign(dx);

            if (dx === 0) {
                for (let y = 0; y <= Math.abs(dy); y++) {
                    let ycr = y * dys + y0;
                    if (this.map[ycr][x0] == -1) return;
                    this.map[ycr][x0] = color;
                }
            } else {
                let derr = Math.abs(dy / dx);
                let error = 0;
                let y = y0;
                for (let x = 0; x < Math.abs(dx); x++) {
                    let xcr = x * dxs + x0;
                    error += derr;
                    let first = true;
                    while (error > 0 || first) {
                        first = false;
                        //console.log(this.map[y][xcr])
                        if (this.map[y][xcr] == -1) return
                        this.map[y][xcr] = color
                        if (error > 0) {
                            y += dys
                            error -= 1;
                        }
                    }
                }
                //this.map[y0][x0] = 8;
                this.map[y1][x1] = color;
            }

        }
        see(x, y, dir, r = 40) {
            let dphi = 1 / r;
            let min_phi = this.dirsA[dir] - 0.523598776;
            let max_phi = this.dirsA[dir] + 0.523598776;
            let dxy = this.dirs[(dir + 2) % 8];
            //console.log(dphi)
            for (let phi = min_phi; phi <= min_phi + 1.04719755; phi += dphi) {

                let ex = Math.round(x + r * Math.cos(phi))
                let ey = Math.round(y + r * Math.sin(phi))
                //console.log(phi, Math.cos(phi),Math.sin(phi), ex,ey)
                this.ray(x, y, ex, ey, 0)
            }
        }
        run(model) {
            let alive = true;
            let count = 0;
            while (alive && count++ < 200) {

                let buf = tf.buffer(this.shape);
                buf.set(2, this.pos[1], this.pos[0]);
                buf.set(1,
                    this.pos[1] + this.dirs[(this.dir + 4) % 8][1],
                    this.pos[0] + this.dirs[(this.dir + 4) % 4][0]
                );

                const actions = model.predict(tf.stack([tf.stack([tf.tensor(this.map), buf.toTensor()])])).buffer();
                const action = actions.values.indexOf(Math.max(...actions.values));
                this.dir = (this.dir + 6 + action * 2) % 8;

                this.pos[0] += this.dirs[this.dir][0];
                this.pos[1] += this.dirs[this.dir][1];

                this.see(this.pos[0], this.pos[1], this.dir)

                let wall = this.map[this.pos];
                //console.log(wall, this.dirs[this.dir])
                if (wall == -1) alive = false;
            }
        }
    }

