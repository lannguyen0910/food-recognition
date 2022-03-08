var messageArea = null,
  wrapperArea = null,
  btnNewPhoto = null,
  btnDownload = null,
  videoCamera = null,
  canvasPhoto = null,
  uploadPhoto = null;

var detectBtn = document.querySelector("#analyze-button");

function clearContent() {
  $('#image-display').empty(); // removes upload img
  $('#upload-label').empty(); //removes upload img's filename
  $('#result-content').remove();   //remove result div (image + labels ...)
}

function runWebcam() {
  
  messageArea = document.querySelector("#upload-label");
  wrapperArea = document.querySelector("#wrapper");

  let video_canvas_html = '<video id="video1" playsinline autoplay></video>' + '<br/>' + '<canvas id="image-canvas"></canvas>';
  $('#webcam-video').html(video_canvas_html);

  btnNewPhoto = document.querySelector("#capture-img");
  btnDownload = document.querySelector("#download-img");
  videoCamera = document.querySelector("video");
  canvasPhoto = document.querySelector("canvas");
  uploadPhoto = document.querySelector("#file-input");
  

  if (window.location.protocol != 'http:' && window.location.protocol != "file:") {
    window.location = /.*redirect=([^&]*).*/.exec(document.location.href)[1];
    window.location.href = 'http:' + window.location.href.substring(window.location.protocol.length);
    return;
  }

  if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
  }

  if (navigator.mediaDevices.getUserMedia === undefined) {
    navigator.mediaDevices.getUserMedia = function (constraints) {
      let getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

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
      clearContent();
      detectBtn.setAttribute('disabled', 'disabled'); //disable detection button

      
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
}

// Capture image on webcam
function takeAPhoto() {
  canvasPhoto.getContext("2d").drawImage(videoCamera, 0, 0, videoCamera.width, videoCamera.height);

  img = document.createElement('img');
  img.id = 'user-image';
  img.src = canvasPhoto.toDataURL('image/jpg', 1.0);
  $('#image-display').prepend(img);

  window.stream.getTracks().forEach(function(track) {
    track.stop();
  });
  $('#webcam-video').empty();

  btnDownload.removeAttribute("disabled");
  detectBtn.removeAttribute("disabled");

}

// Download capture image from webcam to device
function downloadPhoto() {
  canvasPhoto.toBlob(function (blob) {
    let link = document.createElement("a");
    link.download = "photo.jpg";
    link.setAttribute("href", URL.createObjectURL(blob));
    link.dispatchEvent(new MouseEvent("click"));

  }, "image/jpeg", 1);
}


function base64ToBlob(base64, mime) 
{
    mime = mime || '';
    let sliceSize = 1024;
    let byteChars = window.atob(base64);
    let byteArrays = [];

    for (let offset = 0, len = byteChars.length; offset < len; offset += sliceSize) {
        let slice = byteChars.slice(offset, offset + sliceSize);

        let byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }

        let byteArray = new Uint8Array(byteNumbers);

        byteArrays.push(byteArray);
    }

    return new Blob(byteArrays, {type: mime});
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

  const form = $('#file-upload');
  detectBtn.addEventListener('click', function(e) {

    let url = '/analyze';                
    let image = img.src;
    let base64ImageContent = image.replace(/^data:image\/(png|jpg);base64,/, "");
    let blob = base64ToBlob(base64ImageContent, 'image/png');                
    
    let blobFile = document.getElementById('blob-file');
    let file = new File([blob], "img.png",{type:"image/png", lastModified:new Date().getTime()});
    let container = new DataTransfer();

    container.items.add(file);
    blobFile.files = container.files;

    formData = new FormData(form);

    $.ajax({
      url: url,
      data: formData,// the formData function is available in almost all new browsers.
      type:"POST",
      contentType:false,
      processData:false,
      cache:false,
      dataType:"json", // Change this according to your response from the server.
      error:function(err){
          console.error(err);
      },
      success:function(data){
          console.log(data);
      },
      complete:function(){
          console.log("Request finished.");
      }
  });
  },false);
}