var el = x => document.getElementById(x);
var detectBtn = document.querySelector("#analyze-button");

function showPicker() {
  $("#file-input").click();
}

function clearContent() {
  $('#image-display').empty(); // removes upload img
  $('#upload-label').empty(); //removes upload img's filename
  $('#result-content').remove();   //remove result div (image + labels ...)
}

// Show uploaded image or video
function showPicked(input) {

  const extension = input.files[0].name.split(".")[1].toLowerCase();
  const reader = new FileReader();

  reader.onload = function(e) {
    clearContent();
    el("upload-label").innerHTML = input.files[0].name;
    var file_url = e.target.result;

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
  
  };

  detectBtn.removeAttribute("disabled");

  reader.readAsDataURL(input.files[0]);
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