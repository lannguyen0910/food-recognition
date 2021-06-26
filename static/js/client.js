'use strict';
var el = x => document.getElementById(x);

function showPicker() {
  $("#file-input").click();
}

function showPicked(input) {
    console.log(input);
    el("upload-label").innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function(e) {
        console.log(el("image-display"));
        var file_url = e.target.result
        var img_html = '<img id="user-image" src="' + file_url + '" style="display: block;margin-left: auto;margin-right: auto;width: 50%;"/>'
        + '<canvas id="image-canvas"></canvas>';
        $('#image-display').html(img_html); // replaces previous img and canvas
    };
    reader.readAsDataURL(input.files[0]);
}

function clear() {
  $('#image-display').empty(); // removes previous img
  $('#result-content').empty(); 
}

// Enable the webcam
function runWebcam() {
  clear();
  var video_html = '<video playsinline autoplay></video>';
  $('#webcam-video').html(video_html);
  var video = document.querySelector('video');

  var constraints = {
    audio: false,
    video: { width: 9999, height: 9999 },
  };

  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    window.stream = stream; // make stream available to browser console
    video.srcObject = stream;
  }).catch(function(error) {
    console.log(error.message, error.name);
  });

  // Disable image upload
  $('#analyze-button').prop('disabled', true);

  // Reset button
  $('#webcam-btn').removeClass('btn-info').addClass('btn-danger shutter-btn')
    .text('Snap Picture').click(snapImageOnWebcam).off('click', runWebcam);
}

function snapImageOnWebcam() {
  var video = document.querySelector('video');
  $('#image-display').html('<canvas id="image-canvas"></canvas>');
  var canvas = document.querySelector('#image-canvas');

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);


  var img = document.createElement('img');
  img.id = 'user-image';
  img.src = canvas.toDataURL();
  $('#image-display').prepend(img);

  window.stream.getVideoTracks()[0].stop();
  $('#webcam-video').empty();

  // Reset button
  $('#webcam-btn').removeClass('shutter-btn btn-danger').addClass('btn-info')
    .text('New Picture?').off('click', snapImageOnWebcam).click(runWebcam);

  // Re-enable image upload
  $('#analyze-button').prop('disabled', false);
}