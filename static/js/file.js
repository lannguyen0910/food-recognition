var el = x => document.getElementById(x);
var detectBtn = document.querySelector("#analyze-button");
var img;

function showPicker() {
  $("#file-input").click();
}

function clear() {
  $('#image-display').empty(); // removes upload img
  $('#upload-label').empty(); //removes upload img's filename
  $('#result-content').remove();   //remove result div (image + labels ...)
}

// Stop streaming webcam and empty some buttons
// function stopWebcam(){
//   window.stream.getTracks().forEach(function(track) {
//     track.stop();
//   });
//   $('#webcam-video').empty();

//   btnDownload.setAttribute("disabled","disabled");
//   btnNewPhoto.setAttribute("disabled","disabled");
// }

// Upload image or video session
function showPicked(input) {
  // if(window.stream){
  //   stopWebcam();
  // }

  var extension = input.files[0].name.split(".")[1].toLowerCase();
  var reader = new FileReader();

  reader.onload = function(e) {
    clear();
    el("upload-label").innerHTML = input.files[0].name;
    var file_url = e.target.result

    if (extension === "mp4" || extension === 'avi' || extension === '3gpp' || extension === '3gp'){
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

    // $('#webcam-video').prop('disabled', true); //disable webcam when upload file
  
  };

  detectBtn.removeAttribute("disabled");

  reader.readAsDataURL(input.files[0]);
}

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