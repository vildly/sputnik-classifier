import { useState } from "react"
import JSONUploader from "../components/JSONUploader"
import Input from "../components/Input"
import Loading from "../components/Loading"
import { HexColors } from "../lib/colors"
import { sendRaw } from "../services/api"
import Form from "../components/Form"
import Flash from "../components/Flash"

export default function UploadView() {
    const [flash, setFlash] = useState<string>("")
    const [error, setError] = useState<string>("")
    const [processing, setProcessing] = useState<boolean>(false)
    const [value, setValue] = useState<string>("")

    async function handleSubmit(): Promise<void> {
        try {
            // Prevent if no value
            if (!value) return
            // Prevent re-submission
            if (processing) return

            setProcessing(true)
            const res = await sendRaw(value)
            if (res && typeof res === "string") {
                setFlash(res)
            }

            // Reset
            setError("")
            setValue("")
            setProcessing(false)
        } catch(err: any) {
            setError(err.message)
            setProcessing(false)
        }
    }

    return (
        <div className="flex flex-col space-y-2">
            {flash && <Flash message={flash} />}
            {error && <Flash message={error} className="bg-red-900 border-red-400" />}
            <Form onSubmit={handleSubmit} className="flex-row">
                <JSONUploader
                    onChange={setValue}
                    onError={setError}
                    className="w-full"
                />
                {!processing && value && <Input
                    type="submit"
                    value="Submit"
                    className="cursor-pointer"
                />}
            </Form>
            {processing && <Loading color={HexColors.WHITE} className="mx-auto" />}
        </div>
    )
}
