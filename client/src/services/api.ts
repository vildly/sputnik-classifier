import { StrictJSON } from "../lib/json"

const domain = "http://localhost:5000"

function check(res: Response): void {
    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
    }
}

export async function sendRaw(json: string): Promise<string | null> {
    // Validate argument
    if (!StrictJSON.validator(json)) {
        throw new Error("Invalid JSON, must be enclosed by {} and have at least on key")
    }

    const res = await fetch(`${domain}/data`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        // No need to stringify here as it should already be JSON
        body: json
    })

    check(res)

    const contenttype = res.headers.get("content-type")
    if (contenttype && contenttype.match(/text\/plain/)) return res.text()
    return null
}

// TODO: Add return interface here when I know the return type fully
export async function getData(id: string): Promise<any> {
    const res = await fetch(`${domain}/data/${id}`, {
        method: "GET",
        headers: { "content-type": "application/json" },
    })

    check(res)

    const contenttype = res.headers.get("content-type")
    if (contenttype && contenttype.match(/text\/plain/)) return res.text()

    return res.json()
}
