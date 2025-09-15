
function download(filename, text) {
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

function fileTimeToDate( fileTimeBigInt ) {
    const nano100 = Number(BigInt.asUintN(64, fileTimeBigInt)/BigInt(100))
    return new Date ( nano100 / 10000 - 11644473600000 )
}

let prevHeap = 0

function heapChecker(name){
    const currHeap = Module.HEAP8.byteLength
    console.log(`${name} - Heap growth ${Math.floor((currHeap - prevHeap)/1000/1000)} Mo`)
    prevHeap = currHeap
}

class AnalysisMenu{
    load(){
        this.sections = []
        document.querySelectorAll('[ax-open-analysis]').forEach(el=>{
            const target = el.getAttribute('ax-open-analysis')
            if(target == ''){
                el.onclick = () => this.closeAll()
                return
            }
            el.onclick = () => this.open(target)
            this.sections.push(target)
        })
    }

    open(section){
        this.closeAll()
        document.getElementById(section).hidden=false
    }

    closeAll(){
        this.sections.forEach(section=>this.close(section))
    }

    close(section){
        document.getElementById(section).hidden=true
    }

    addSection(section){
        this.sections.push(section)
    }

}