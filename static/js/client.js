
var el = x => document.getElementById(x);

function showPicker() {
  $("#file-input").click();
}

function clear() {
  $('#image-display').empty(); // removes previous img
  $('#upload-label').empty(); //removes previous img name
  $('#result-content').empty();   //remove result content (image + labels ...)
  $('canvas').empty();
}

function showPicked(input) {
  clear();
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
      var file_url = e.target.result
      var img_html = '<img id="user-image" src="' + file_url + '" style="display: block;margin-left: auto;margin-right: auto;width: 50%;"/>';
      $('#image-display').html(img_html); // replaces previous img and canvas
  };
  reader.readAsDataURL(input.files[0]);
}

var messageArea = null,
  wrapperArea = null,
  btnNewPhoto = null,
  btnDownload = null,
  videoCamera = null,
  canvasPhoto = null,
  uploadPhoto = null;


function runWebcam() {
  clear();
  messageArea = document.querySelector("#upload-label");
  wrapperArea = document.querySelector("#wrapper");

  var video_canvas_html = '<video id="video1" playsinline autoplay></video>' + '<br/>' + '<canvas id="image-canvas"></canvas>';
  $('#webcam-video').html(video_canvas_html);

  btnNewPhoto = document.querySelector("#capture-img");
  btnDownload = document.querySelector("#download-img");
  videoCamera = document.querySelector("video");
  canvasPhoto = document.querySelector("canvas");
  uploadPhoto = document.querySelector("#file-input");
  

  if (window.location.protocol != 'https:' && window.location.protocol != "file:") {
    window.location.href = 'https:' + window.location.href.substring(window.location.protocol.length);
    return;
  }

  if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
  }

  if (navigator.mediaDevices.getUserMedia === undefined) {
    navigator.mediaDevices.getUserMedia = function (constraints) {

      var getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

      if (!getUserMedia) {
        return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
      }

      return new Promise(function (resolve, reject) {
        getUserMedia.call(navigator, constraints, resolve, reject);
      });
    };
  }

  navigator.mediaDevices.getUserMedia({
    video: true
  })
    .then(function (stream) {
      if ("srcObject" in videoCamera) {
        videoCamera.srcObject = stream;
      } else {
        videoCamera.src = window.URL.createObjectURL(stream);
      }

      $('#analyze-button').prop('disabled', true); //disable detection
      $('#file-input').prop('disabled', true); //disable image upload

      messageArea.style.display = "none";
      wrapperArea.style.display = "block";
      
      btnNewPhoto.removeAttribute("disabled");
      uploadPhoto.setAttribute("disabled","disabled");


      btnNewPhoto.onclick = takeAPhoto;
      btnDownload.onclick = downloadPhoto;
      

      videoCamera.onloadedmetadata = function () {
        videoCamera.setAttribute("width", this.videoWidth);
        videoCamera.setAttribute("height", this.videoHeight);
        canvasPhoto.setAttribute("width", this.videoWidth);
        canvasPhoto.setAttribute("height", this.videoHeight);
        videoCamera.play();
        
      };
    })
    .catch(function (err) {
      messageArea.innerHTML = err.name + ": " + err.message;
    });
};

function takeAPhoto() {
  canvasPhoto.getContext("2d").drawImage(videoCamera, 0, 0, videoCamera.width, videoCamera.height);
  btnDownload.removeAttribute("disabled");
  $('#analyze-button').prop('disabled', false); //disable image upload

};

function downloadPhoto() {
  canvasPhoto.toBlob(function (blob) {
    var link = document.createElement("a");
    link.download = "photo.jpg";
    link.setAttribute("href", URL.createObjectURL(blob));
    link.dispatchEvent(new MouseEvent("click"));

  }, "image/jpeg", 1);
};