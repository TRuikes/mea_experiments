class Loading{
    constructor(id){
        this.div = document.getElementById(id)
        this.div.hidden = true
        this.loadingMax = 257
        this.maxWidth = 100
        this.loading = 0        
    }

    reset(max){
        this.loading = 0
        this.loadingMax = max
        this.refresh()
    }

    add(){
        this.loading++
        this.refresh()
    }

    refresh(){
        this.div.hidden = (this.loading >= this.loadingMax)
        // console.log(this.loading, this.loadingMax)
        this.div.style.width = `${this.loading / this.loadingMax * this.maxWidth}%`
    }
}

function showLoading(id){
    const div = document.getElementById(id)
    div.innerText = 'Loading...'
    div.hidden = false
}

function hideLoading(id){
    const div = document.getElementById(id)
    div.hidden = true
    div.innerText = ''
}