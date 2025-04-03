import type React from "react"

import { useState } from "react"
import { Plus, Trash2, AlertCircle } from "lucide-react"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"

// URL validation schema
const urlSchema = z.string().url("Please enter a valid URL")

export interface UrlManagerProps {
  urls: string[]
  onUrlsChange: (urls: string[]) => void
}

export default function UrlManager({ urls, onUrlsChange }: UrlManagerProps) {
  const [inputValue, setInputValue] = useState("")
  const [open, setOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const validateAndAddUrl = () => {
    if (!inputValue.trim()) return

    // Reset previous error
    setError(null)

    try {
      // Validate URL format
      urlSchema.parse(inputValue)

      // Check for duplicates
      if (urls.includes(inputValue)) {
        setError("This URL already exists in the list")
        return
      }

      // Add URL to the list
      onUrlsChange([...urls, inputValue])
      setInputValue("")
    } catch (err) {
      if (err instanceof z.ZodError) {
        setError(err.errors[0].message)
      } else {
        setError("An error occurred while validating the URL")
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      validateAndAddUrl()
    }
  }

  const handleDeleteUrl = (indexToDelete: number) => {
    onUrlsChange(urls.filter((_, index) => index !== indexToDelete))
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="h-8 w-8">
          <Plus className="h-4 w-4" />
          <span className="sr-only">Manage URLs</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>URL Manager</DialogTitle>
        </DialogHeader>
        <div className="flex items-center space-x-2">
          <Input
            placeholder="Enter URL (e.g., https://example.com)"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value)
              setError(null) // Clear error when input changes
            }}
            onKeyDown={handleKeyDown}
            className={`flex-1 ${error ? "border-destructive" : ""}`}
          />
          <Button size="icon" onClick={validateAndAddUrl}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        {error && (
          <Alert variant="destructive" className="mt-2">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="mt-4 max-h-[200px] overflow-y-auto">
          {urls.length > 0 ? (
            <ul className="space-y-2">
              {urls.map((url, index) => (
                <li key={index} className="rounded-md border p-2 text-sm break-all flex items-center justify-between">
                  <span className="mr-2">{url}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleDeleteUrl(index)}
                    className="h-8 w-8 text-destructive hover:bg-destructive/10"
                  >
                    <Trash2 className="h-4 w-4" />
                    <span className="sr-only">Delete</span>
                  </Button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-center text-sm text-muted-foreground">No URLs added yet</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}