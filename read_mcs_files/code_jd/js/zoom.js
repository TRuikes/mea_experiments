class Zoom{
    constructor(){
        this.reset()
    }

    reset(){
        this.active = false
        this.current = {
            x0:0,
            x1:1,
            y0:0,
            y1:1
        }
        this.previous = {}
        this.copyInPrevious()
    }

    down(x0,y0){
        this.copyInPrevious()
        this.current.x0 = x0
        this.current.y0 = y0
    }

    up(x1,y1){
        this.current.x0 = this.Xpos(this.current.x0)
        this.current.y0 = this.Ypos(this.current.y0)
        this.current.x1 = this.Xpos(x1)
        this.current.y1 = this.Ypos(y1)
        this.active = true
        this.copyInPrevious()
    }

    Xpos(x){
        return x * (this.previous.x1 - this.previous.x0) + this.previous.x0 
    }
    
    Ypos(y){
        return y * (this.previous.y1 - this.previous.y0) + this.previous.y0
    }

    copyInPrevious(){
        this.previous.x0 = this.current.x0
        this.previous.x1 = this.current.x1
        this.previous.y0 = this.current.y0
        this.previous.y1 = this.current.y1
    }

    getX(px){
        if(!zoom.active) return px
        return (px - zoom.x0) / (zoom.x1-zoom.x0)
    }

    get x0(){
        return this.current.x0
    }

    get x1(){
        return this.current.x1
    }

    get y0(){
        return this.current.y0
    }

    get y1(){
        return this.current.y1
    }
}