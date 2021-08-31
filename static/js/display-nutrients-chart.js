var csv_file = document.querySelector("#csv_file");
var csv_file2 = document.querySelector("#csv_file2");
var canvas1 = document.querySelector("#container1");
var canvas2 = document.querySelector("#container2");

var filename =  csv_file.textContent;
var filename2 = csv_file2.textContent;

var TITLE = 'Nutrients Statistics Chart';

// `false` for vertical column chart, `true` for horizontal bar chart
var HORIZONTAL = false;

// `false` for individual bars, `true` for stacked bars
var STACKED = true;  


var LABELS = 'names';  

// For each column representing a data series, define its name and color
var SERIES = [  
    {
        column: 'calories',
        name: 'calories',
        color: 'red'
    },
    {
        column: 'protein',
        name: 'protein',
        color: 'green'
    },
    {
        column: 'fat',
        name: 'fat',
        color: 'blue'
    },
    {
        column: 'carbs',
        name: 'carbs',
        color: 'yellow'
    },
    {
        column: 'fiber',
        name: 'fiber',
        color: 'purple'
    }
];

// x-axis label and label in tooltip
var X_AXIS = 'Elements Info';

// y-axis label, label in tooltip
var Y_AXIS = 'Amount';

// `true` to show the grid, `false` to hide
var SHOW_GRID = true; 

// `true` to show the legend, `false` to hide
var SHOW_LEGEND = true; 

function getChart(val){
    if(val.value === "1"){
        $('#display-chart').empty();
        canvas1.style.display = "block";
        canvas2.style.display = 'none'

        // Read csv, bypass browser cache, and create chart
        $.get(filename, {'_': $.now()}, function(csvString) {

            let rows = Papa.parse(csvString, {header: true, skipEmptyLines: true}).data;

            let datasets = SERIES.map(function(el) {
                return {
                    label: el.name,
                    labelDirty: el.column,
                    backgroundColor: el.color,
                    data: []
                }
            });

            rows.map(function(row) {
                datasets.map(function(d) {
                    d.data.push(row[d.labelDirty])
                })
            });

            let barChartData = {
                labels: rows.map(function(el) { return el[LABELS] }),
                datasets: datasets
            };

            let ctx = document.getElementById('container1').getContext('2d');

            new Chart(ctx, {
            type: HORIZONTAL ? 'horizontalBar' : 'bar',
            data: barChartData,
            
            options: {
                title: {
                    display: true,
                    text: TITLE,
                    fontSize: 20,
                },
                legend: {
                    display: SHOW_LEGEND,
                },
                scales: {
                xAxes: [{
                    stacked: STACKED,
                    scaleLabel: {
                        display: X_AXIS !== '',
                        labelString: X_AXIS
                    },
                    gridLines: {
                        display: SHOW_GRID,
                    },
                    ticks: {
                    beginAtZero: true,
                    callback: function(value, index, values) {
                        return value.toLocaleString();
                    }
                    }
                }],
                yAxes: [{
                    stacked: STACKED,
                    beginAtZero: true,
                    scaleLabel: {
                        display: Y_AXIS !== '',
                        labelString: Y_AXIS
                    },
                    gridLines: {
                        display: SHOW_GRID,
                    },
                    ticks: {
                        beginAtZero: true,
                        callback: function(value, index, values) {
                            return value.toLocaleString()
                        }
                    }
                }]
                },
                tooltips: {
                displayColors: false,
                callbacks: {
                    label: function(tooltipItem, all) {
                    return all.datasets[tooltipItem.datasetIndex].label
                        + ': ' + tooltipItem.yLabel.toLocaleString();
                    }
                }
                }
            }
            });
        });
    }
    else {
        $('#display-chart').empty();
        canvas2.style.display = "block";
        canvas1.style.display = 'none'
        $.get(filename2, {'_': $.now()}, function(csvString) {
            let rows = Papa.parse(csvString, {header: true, skipEmptyLines: true}).data;

            let names = Object.keys(rows[0]).slice(1);

            let SERIES1 = [];
            let info = ['calories', 'protein', 'fat', 'carbs', 'fiber'];

            for(let i=0;i<names.length;i++){
                let randomColor = Math.floor(Math.random()*16777215).toString(16);
                let colour = "#" + randomColor;

                SERIES1.push({
                    column: names[i],
                    name: names[i],
                    color: colour
                });
            }

            let datasets = SERIES1.map(function(el) {
                return {
                    label: el.name,
                    labelDirty: el.column,
                    backgroundColor: el.color,
                    data: []
                }
            });


            rows.map(function(row) {
                datasets.map(function(d) {
                    d.data.push(row[d.labelDirty])
                })
            });

            let barChartData = {
                labels: info,
                datasets: datasets
            };

            var ctx = document.getElementById('container2').getContext('2d');

            new Chart(ctx, {
                type: HORIZONTAL ? 'horizontalBar' : 'bar',
                data: barChartData,
                
                options: {
                    title: {
                        display: true,
                        text: TITLE,
                        fontSize: 20,
                    },
                    legend: {
                        display: SHOW_LEGEND,
                    },
                    scales: {
                    xAxes: [{
                        stacked: STACKED,
                        scaleLabel: {
                            display: X_AXIS !== '',
                            labelString: X_AXIS
                        },
                        gridLines: {
                            display: SHOW_GRID,
                        },
                        ticks: {
                        beginAtZero: true,
                        callback: function(value, index, values) {
                            return value.toLocaleString();
                        }
                        }
                    }],
                    yAxes: [{
                        stacked: STACKED,
                        beginAtZero: true,
                        scaleLabel: {
                            display: Y_AXIS !== '',
                            labelString: Y_AXIS
                        },
                        gridLines: {
                            display: SHOW_GRID,
                        },
                        ticks: {
                            beginAtZero: true,
                            callback: function(value, index, values) {
                                return value.toLocaleString()
                            }
                        }
                    }]
                    },
                    tooltips: {
                    displayColors: false,
                    callbacks: {
                        label: function(tooltipItem, all) {
                        return all.datasets[tooltipItem.datasetIndex].label
                            + ': ' + tooltipItem.yLabel.toLocaleString();
                        }
                    }
                    }
                }
            });

        });
    }
}