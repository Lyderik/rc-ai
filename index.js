require('@tensorflow/tfjs-node-gpu');  // Use '@tensorflow/tfjs-node-gpu' if running with GPU.

const tf = require('@tensorflow/tfjs');

// Load the binding:

gm = require('./game')



const model = tf.sequential();

/*model.add(tf.layers.conv2d({
  inputShape: [120, 80],
  activation: 'relu',
  units: 100,
  filters: 100
}));*/

model.add(tf.layers.flatten({
  inputShape: [2, 80, 120],
  units: 160,
  activation: "relu"
}));

model.add(tf.layers.dense({
  units: 80,
  activation: "relu"
}));


model.add(tf.layers.dense({
  units: 20,
  activation: "relu"
}))

model.add(tf.layers.dense({
  units: 3,
  activation: "softmax"
}))


model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

game = new gm();
game.init().then(() => {
  //;
  /*game.ray(2,7,10,25,5);
  game.ray(2,6,10,12,2);
  game.ray(18,4,2,5,3);
  game.ray(18,25,2,3,4);
  game.ray(18,25,18,2,6);
  game.ray(1,1,18,1,7);*/
  /*game.see(2,15,0);
  game.see(18,18,4);
  game.ray(18,18,8,20,2)*/
  game.run(model)
  console.table(game.map, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
  console.table(game.map, [40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79])
  console.table(game.map, [80, 81, 82, 83, 85, 86, 87, 88, 89, 80, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
}).catch((err) => {
  console.log(err);
});
/*
const xs = tf.randomNormal([10,120,80]);
const ys = tf.randomUniform([10,120, 8],0,1);


model.fit(xs, ys, {
    epochs: 1000,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        console.log(`Epoch ${epoch}: loss = ${log.loss}`);
      }
    }
  });
*/