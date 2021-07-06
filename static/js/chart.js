var csv_file = document.querySelector("#csv_file");
var filename =  csv_file.textContent;
console.log("Filename: ", filename);

var result = document.querySelector('#image-result');
async function getData(){
    const response = await fetch(filename);
    const data = await response.text();
    console.log("Response: ", response);

    return data;
}
// const response = fetch(filename);
// console.log("Response: ", response);
async function setUp(){
    const data = await getData();
    console.log("Data: ", data);

    var attrNames = data.split('\n').slice(0, 1);
    var headers = [];
    attrNames.forEach(attr => {
        var header = attr.split(',').slice(3);
        for(let i = 0; i < header.length;i++){
            headers.push(header[i]);
        }

    })
    // var attrNames = data.split('\n').slice(0, 1).split(',').slice(7);
    console.log("attr names: ", attrNames);
    console.log("headers: ", headers);

    var rows = data.split('\n').slice(1, -1);
    console.log("rows: ", rows);

    var rowsValues=[], 
    rowValues = [], 
    labelNames = [], 
    colors = [];

    var numberOfAttributes = 0;

    rows.forEach(row => {
        let color = [];

        var labelName = row.split(',').slice(2,3);
        labelNames.push(labelName);

        var cols = row.split(',').slice(3);
        for (let i=0; i < cols.length; i++){
            rowValues.push(cols[i]);

            let randomColor = Math.floor(Math.random()*16777215).toString(16);
            let colour = "#" + randomColor;
            color.push(colour);
        }
        rowsValues.push(rowValues);
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
            labels: headers,
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
}

window.addEventListener('load', setUp);