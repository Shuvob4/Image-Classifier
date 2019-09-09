const MOBILENET_MODEL_PATH =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';


$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop("files")[0];
    reader.readAsDataURL(file);
}); 

let model;
(async function () {
    model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    console.log('Sucessfully loaded model');
    $(".progress-bar").hide();
})();

function preprocessImage(image) {

  // resize the input image to mobilenet's target size of (224, 224)
  let tensor = tf.browser.fromPixels(image)
    .resizeNearestNeighbor([224, 224])
    .toFloat();


  let offset = tf.scalar(127.5);
    return tensor.sub(offset)
      .div(offset)
      .expandDims();
  
}


$("#predict_button").click(async function () {
  let image = document.getElementById("selected-image");
  let tensor = preprocessImage(image);
      

  let predictions = await model.predict(tensor).data();
  let top5 = Array.from(predictions)
      .map(function (p, i) {
          return {
              probability: p,
              className: IMAGENET_CLASSES[i]
          };
      }).sort(function (a, b) {
          return b.probability - a.probability;
      }).slice(0, 5);

  $("#prediction-list").empty();
  top.forEach(function (p) {
      $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
  });

});


