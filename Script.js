function toggleMenu() {
    var navMenu = document.getElementById("navMenu");
    if (navMenu.style.display === "block") {
        navMenu.style.display = "none";
    } else {
        navMenu.style.display = "block";
    }
}
function runPythonCode() {
    $.ajax({
        type: "POST",
        url: "run_python_code.py",
        success: function (response) {
            $("#pythonOutput").html(response);
        }
    });
}
function runPythonCode() {
    $.ajax({
        type: "POST",
        url: "run_python_code.py",
        success: function (response) {
            $("#pythonOutput").html(response);
        }
    });
}
