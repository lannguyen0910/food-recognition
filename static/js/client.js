var el = x => document.getElementById(x);
var detectBtn = document.querySelector("#analyze-button");

function showPicker() {
  $("#file-input").click();
}

function clear() {
  $('#image-display').empty(); // removes upload img
  $('#upload-label').empty(); //removes upload img's filename
  $('#result-content').remove();   //remove result div (image + labels ...)
}

// Stop streaming webcam and empty some buttons
function stopWebcam(){
  window.stream.getTracks().forEach(function(track) {
    track.stop();
  });
  $('#webcam-video').empty();

  btnDownload.setAttribute("disabled","disabled");
  btnNewPhoto.setAttribute("disabled","disabled");
}

// Upload image or video session
function showPicked(input) {
  if(window.stream){
    stopWebcam();
  }

  var extension = input.files[0].name.split(".")[1].toLowerCase();
  var reader = new FileReader();

  reader.onload = function(e) {
    clear();
    el("upload-label").innerHTML = input.files[0].name;
    var file_url = e.target.result

    if (extension === "mp4" || extension === 'avi' || extension === '3gp'){
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

    $('#webcam-video').prop('disabled', true); //disable webcam when upload file
  
  };

  detectBtn.removeAttribute("disabled");

  reader.readAsDataURL(input.files[0]);
}

// Webcam session
var messageArea = null,
  wrapperArea = null,
  btnNewPhoto = null,
  btnDownload = null,
  videoCamera = null,
  canvasPhoto = null,
  uploadPhoto = null;


function runWebcam() {
  
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
      clear();
      detectBtn.setAttribute('disabled', 'disabled'); //disable detection

      
      messageArea.style.display = 'block';
      wrapperArea.style.display = "block";
      canvasPhoto.style.display = "none";
      
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

// Capture image on webcam
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

  // var timestamp = new Date().getTime().toString();
  // messageArea.innerHTML = timestamp +'.png';
  btnDownload.removeAttribute("disabled");

};

// Download capture image from webcam to device
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

// function imageExists(image_url){ //https://stackoverflow.com/questions/18837735/check-if-image-exists-on-server-using-javascript
//     // var http = new XMLHttpRequest();

//     // http.open('GET', image_url, false);
//     // http.setRequestHeader("Access-Control-Allow-Origin", "*");
//     // http.setRequestHeader('Access-Control-Allow-Credentials', 'true');
//     // http.send();

//     // return http.status != 404;
//     const myRequest = new Request(image_url, {
//       method: 'HEAD',
//       headers: {
//           "Content-Type": "text/plain"
//       },
//       mode: 'cors',
//       cache: 'default',
//     });

//     fetch(myRequest)
//         .then(function (response) {
//             if (response.ok) {
//                 return true;
//             } else {
//                 return false;
//             }
//             }).catch(function(err) {
//                 alert(err);
//             });

// }

// // youtube video url
// function YouTubeGetID(url){ //https://gist.github.com/takien/4077195
//   var ID = '';
//   url = url.replace(/(>|<)/gi,'').split(/(vi\/|v=|\/v\/|youtu\.be\/|\/embed\/)/);
//   if(url[2] !== undefined) {
//     ID = url[2].split(/[^0-9a-z_\-]/i);
//     ID = ID[0];
//   }
  
//   else {
//     ID = url;
//   }
  
//   return ID;
// }

// function validVideoId(id) { //https://gist.github.com/tonY1883/a3b85925081688de569b779b4657439b
// 		var img = new Image();
//     var load = null;
// 		img.src = "http://img.youtube.com/vi/" + id + "/mqdefault.jpg";
// 		img.onload = function () {
// 			load = checkThumbnail(this.width);
      
//       return load;
//     }
// 	}

// function checkThumbnail(width) {
//   //HACK a mq thumbnail has width of 320.
//   //if the video does not exist(therefore thumbnail don't exist), a default thumbnail of 120 width is returned.
//   if (width === 120) {
//     return false;
//   }
//   return true;
// }

// Check input URL wether it has 'http' or 'https' protocol
function trackURL(url){
  if (isValidHttpUrl(url)){
    detectURLBtn.removeAttribute("disabled");
  }
  else{
    detectURLBtn.setAttribute("disabled","disabled");
  }
}

// Click to close alert notification
function closeAlert(){
  $('#alert').remove();
}

// Send image source from client to server under base64 format
// var sendBase64ToServer = function(base64){
//     var httpPost = new XMLHttpRequest(),
//         path = "/",
//         data = JSON.stringify({image: base64});
//         console.log(data);
//     httpPost.onreadystatechange = function(err) {
//             if (httpPost.readyState == 4 && httpPost.status == 200){
//                 console.log(httpPost.responseText);
//             } else {
//                 console.log(err);
//             }
//         };
//     // Set the content type of the request to json since that's what's being sent
//     httpPost.open("POST", path, true);
//     httpPost.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
//     httpPost.send(data);
// };

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