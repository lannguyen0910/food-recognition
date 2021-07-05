var csv_file = document.querySelector("#csv_file");
var filename = '/static/csv/' +  csv_file.text();
console.log("Filename: ", filename);

var result = document.querySelector('#image-result');
const response = fetch(filename);
console.log("Response: ", response);

const data = response.text();
var attrNames = data.split('\n').slice(0, 1).split(',').slice(7);
console.log("attr names: ", attrNames);

var rows = data.split('\n').slice(1);
console.log("rows: ", rows);

var rowsValues, rowValues, labelNames, colors = [], [], [], [];
var numberOfAttributes = 0;

rows.forEach(row => {
    let color = [];

    var labelName = row.split(',').slice(6,7);
    labelNames.push(labelName);
    

    var cols = row.split(',').slice(7);
    cols = Array.from(cols, item => item || 0);
    for (let i=0; i < cols.length; i++){
        rowValues.push(cols[i]);

        let randomColor = Math.floor(Math.random()*16777215).toString(16);
        let colour = "#" + randomColor;
        color.push(colour);
    }
    rowsValues.push(val);
    colors.push(color);

    var canvas = document.createElement("canvas");
    canvas.setAttribute("id", "myChart" + numberOfAttributes);
    numberOfAttributes ++;
    result.appendChild(canvas);
});

console.log("labels: ", labelNames);
console.log("rows value: ", rowsValues);

//loop array of data and create chart for each row
rowsValues.forEach(function(e,i){
  var chartID = "myChart"+ i
  var ctx = document.getElementById(chartID).getContext('2d');
  var myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: attrNames,
        datasets: [{
            label: labelNames[i],
            data: e,
            backgroundColor: colors[i],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero:true
                }
            }]
        }
    }
});
})