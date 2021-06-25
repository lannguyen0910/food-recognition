var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
    el("upload-label").innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function(e) {
        el("image-picked").src = e.target.result;
        el("image-picked").className = "";
    };
    reader.readAsDataURL(input.files[0]);
}