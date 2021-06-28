var el = x => document.getElementById(x);
var detectBtn = document.querySelector("#analyze-button");


function showPicker() {
  $("#file-input").click();
}

function clear() {
  $('#image-display').empty(); // removes previous img
  $('#upload-label').empty(); //removes previous img name
  $('#result-content').empty();   //remove result content (image + labels ...)
  // let canvas = document.querySelector("canvas");
  // const context = canvas.getContext('2d');
  // context.clearRect(0, 0, canvas.width, canvas.height);
}

function stopWebcam(){
  window.stream.getTracks().forEach(function(track) {
    track.stop();
  });
  $('#webcam-video').empty();

  btnDownload.setAttribute("disabled","disabled");
  btnNewPhoto.setAttribute("disabled","disabled");
}

// Upload file 
function showPicked(input) {
  clear();
  if(window.stream){
    stopWebcam();
  }
  el("upload-label").innerHTML = input.files[0].name;
  var extension = input.files[0].name.split(".")[1].toLowerCase();
  var reader = new FileReader();
  reader.onload = function(e) {
    var file_url = e.target.result
    if (extension === "mp4"){
      var video_html = '<video autoplay id="user-video" controls> <source id="user-source"></source></video>'
      $('#image-display').html(video_html); // replaces previous video
      var video = el("user-video");
      var source = el("user-source");
      source.setAttribute("src", file_url);
      video.load();
      video.play();
    }

    else if(extension === "jpg" || extension === "jpeg" || extension === "png"){
      var img_html = '<img id="user-image" src="' + file_url + '" style="display: block;margin-left: auto;margin-right: auto;width: 640px; height: 480px"/>';
      $('#image-display').html(img_html); // replaces previous img
    }

    $('#webcam-video').prop('disabled', true); //disable image upload
    
  };
  detectBtn.removeAttribute("disabled");
  reader.readAsDataURL(input.files[0]);
}


// Webcam
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
        window.stream = stream;
      } else {
        videoCamera.src = window.URL.createObjectURL(stream);
      }

      detectBtn.setAttribute('disabled', 'disabled'); //disable detection
      
      messageArea.style.display = 'block';
      wrapperArea.style.display = "block";
      
      btnNewPhoto.removeAttribute("disabled");
      btnDownload.setAttribute("disabled","disabled");


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

  var img = document.createElement('img');
  img.id = 'user-image';
  img.src = canvasPhoto.toDataURL();
  $('#image-display').prepend(img);

  window.stream.getTracks().forEach(function(track) {
    track.stop();
  });
  $('#webcam-video').empty();

  var timestamp = new Date().getTime().toString();
  messageArea.innerHTML = timestamp +'.png';
  btnDownload.removeAttribute("disabled");
  detectBtn.removeAttribute("disabled"); //enable detect

};

function downloadPhoto() {
  canvasPhoto.toBlob(function (blob) {
    var link = document.createElement("a");
    link.download = "photo.jpg";
    link.setAttribute("href", URL.createObjectURL(blob));
    link.dispatchEvent(new MouseEvent("click"));

  }, "image/jpeg", 1);
};


// URL handle

// image url
var detectURLBtn = document.querySelector("#url-button");

function isValidHttpUrl(string) {
  let url;
  
  try {
    url = new URL(string);
  } catch (_) {
    return false;  
  }

  return (url.protocol === "http:" || url.protocol === "https:");
}

function imageExists(image_url){ //https://stackoverflow.com/questions/18837735/check-if-image-exists-on-server-using-javascript
    var http = new XMLHttpRequest();

    http.open('HEAD', image_url, false);
    http.send();

    return http.status != 404;
}

// youtube url
function YouTubeGetID(url){ //https://gist.github.com/takien/4077195
  var ID = '';
  url = url.replace(/(>|<)/gi,'').split(/(vi\/|v=|\/v\/|youtu\.be\/|\/embed\/)/);
  if(url[2] !== undefined) {
    ID = url[2].split(/[^0-9a-z_\-]/i);
    ID = ID[0];
  }
  
  else {
    ID = url;
  }
  
  return ID;
}

function validVideoId(id) { //https://gist.github.com/tonY1883/a3b85925081688de569b779b4657439b
		var img = new Image();
    var load = null;
		img.src = "http://img.youtube.com/vi/" + id + "/mqdefault.jpg";
		img.onload = function () {
			load = checkThumbnail(this.width);
      
      return load;
    }
	}

function checkThumbnail(width) {
  //HACK a mq thumbnail has width of 320.
  //if the video does not exist(therefore thumbnail don't exist), a default thumbnail of 120 width is returned.
  if (width === 120) {
    return false;
  }
  return true;
}

function trackURL(url){
  if (isValidHttpUrl(url) && (imageExists(url) || validVideoId(YouTubeGetID(url)))){
    detectURLBtn.removeAttribute("disabled");
  }
  else{
    detectURLBtn.setAttribute("disabled","disabled");
  }
}

window.onload = function(){

  $('#threshold-range').on('input', function() {
    $('#threshold-text span').html(this.value);
    threshold = $('#threshold-range').val() / 100;
  });

  $('#confidence-range').on('input', function() {
    $('#confidence-text span').html(this.value);
    confidence = $('#confidence-range').val() / 100;
  });
}