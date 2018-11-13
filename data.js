const tf = require('@tensorflow/tfjs');
const fs = require('fs');
PNG = require('pngjs').PNG;

var exports = module.exports = {};

//console.log(png)
exports.import = (filename) => {
    return new Promise((resolve, reject) => {
        fs.createReadStream(filename)
            .pipe(new PNG({colorType: 0}))
            .on('parsed', function(data) {
                let npix = this.width * this.height;
                rbg = [npix];
                for (let i = 0; i < npix; i++) {
                    rbg[i] = (data.readUInt8(i * 4) == 0)?-1:1;;
                }
                resolve({map:rbg, shape :[this.height,this.width]});
            });
    })
};



